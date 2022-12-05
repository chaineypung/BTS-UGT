import argparse
import os
import pathlib
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import yaml
from monai.data import decollate_batch
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from config import get_config
from dataset.brats import get_datasets
from loss import EDiceLoss
from loss import MEEP_KL_Loss
from loss import WEDiceLoss
from loss.dice import EDiceLoss_Val
from utils import AverageMeter, ProgressMeter, save_checkpoint, reload_ckpt_bis, \
    count_parameters, save_metrics, save_args_1, inference, post_trans, dice_metric, \
    dice_metric_batch
from vtunet.vision_transformer import VTUNet as ViT_seg
from vtunet.unetr import UNETR as TransUnet
import warnings
import math
import SimpleITK as sitk
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


# set num GPU
NO_GPU = 0
parser = argparse.ArgumentParser(description='VTUNET BRATS 2021 Training')

# DO not use data_aug argument this argument!!
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=350, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
parser.add_argument('--devices', default='1', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--val', default=1, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")
parser.add_argument('--num_classes', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default="configs/vt_unet_base.yaml", metavar="FILE",
                    help='path to config file', )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', default=False, type=bool, help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')


def main(args):
    # setup
    ngpus = torch.cuda.device_count()
    print(f"Working with {ngpus} GPUs")

    args.exp_name = "logs_base"
    args.save_folder_1 = pathlib.Path(f"./runs/{args.exp_name}/model_1")
    args.save_folder_1.mkdir(parents=True, exist_ok=True)
    args.seg_folder_1 = args.save_folder_1 / "segs"
    args.seg_folder_1.mkdir(parents=True, exist_ok=True)
    args.save_folder_1 = args.save_folder_1.resolve()
    save_args_1(args)
    t_writer_1 = SummaryWriter(str(args.save_folder_1))
    args.checkpoint_folder = pathlib.Path(f"./runs/{args.exp_name}/model_1")

    # Create model
    with open(args.cfg, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    config = get_config(args)

    model_1 = ViT_seg(config, num_classes=args.num_classes,
                      embed_dim=yaml_cfg.get("MODEL").get("SWIN").get("EMBED_DIM"),
                      win_size=yaml_cfg.get("MODEL").get("SWIN").get("WINDOW_SIZE")).cuda(device=NO_GPU)

    model_1.load_from(config)

    # model_1 = TransUnet(img_shape=(64, 64, 64)).cuda(device=NO_GPU)


    if args.resume:
        args.checkpoint = args.checkpoint_folder / "model_best.pth.tar"
        reload_ckpt_bis(args.checkpoint, model_1)

    print(f"total number of trainable parameters {count_parameters(model_1)}")

    model_1 = model_1.cuda(device=NO_GPU)

    model_file = args.save_folder_1 / "model.txt"
    with model_file.open("w") as f:
        print(model_1, file=f)

    criterion = EDiceLoss().cuda(device=NO_GPU)
    criterion_1 = torch.nn.KLDivLoss().cuda(device=NO_GPU)
    criterion_2 = MEEP_KL_Loss().cuda(device=NO_GPU)
    criterian_val = EDiceLoss_Val().cuda(device=NO_GPU)
    metric = criterian_val.metric
    print(metric)
    params = model_1.parameters()


    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(params, lr=args.lr)

    full_train_dataset, l_val_dataset, bench_dataset = get_datasets(args.seed, fold_number=args.fold, normalisation="zscore")#####################################
    train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(l_val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=args.workers)
    bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=args.workers)

    print("Train dataset number of batch:", len(train_loader))
    print("Val dataset number of batch:", len(val_loader))
    print("Bench Test dataset number of batch:", len(bench_loader))

    # Actual Train loop
    best_1 = 0.0
    patients_perf = []

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    t = 10  # warmup
    T = 350  # 共有120个epoch，则用于cosine rate的一共有110个epoch
    n_t = 0.5
    lambda1 = lambda epoch: (0.9 * epoch / t + 0.1) if epoch < t else 0.1 if n_t * (
                1 + math.cos(math.pi * (epoch - t) / (T - t))) < 0.1 else n_t * (
                1 + math.cos(math.pi * (epoch - t) / (T - t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    print("start training now!")

    for epoch in range(args.epochs):
        try:
            # do_epoch for one epoch
            ts = time.perf_counter()

            # Setup
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses_ = AverageMeter('Loss', ':.4e')

            mode = "train" if model_1.training else "val"
            batch_per_epoch = len(train_loader)
            progress = ProgressMeter(
                batch_per_epoch,
                [batch_time, data_time, losses_],
                prefix=f"{mode} Epoch: [{epoch}]")

            end = time.perf_counter()
            metrics = []

            for i, batch in enumerate(zip(train_loader)):
                torch.cuda.empty_cache()
                # measure data loading time
                data_time.update(time.perf_counter() - end)

                inputs_S1, labels_S1 = batch[0]["image"].float(), batch[0]["label"].float()

                inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
                inputs_S1, labels_S1 = inputs_S1.cuda(device=NO_GPU), labels_S1.cuda(device=NO_GPU)

                optimizer.zero_grad()

                segs_S1, uncer = model_1(inputs_S1)

                loss_ = criterion(segs_S1, labels_S1)
                loss_1 = criterion_1(uncer, labels_S1)
                loss_2 = criterion_2(uncer, labels_S1)
                loss_3 = criterion(uncer, labels_S1)

                loss_ = loss_ + 0.0 * loss_1 + 0.2 * loss_2 + 0.2 * loss_3

                t_writer_1.add_scalar(f"Loss/{mode}{''}",
                                      loss_.item(),
                                      global_step=batch_per_epoch * epoch + i)

                # measure accuracy and record loss_
                if not np.isnan(loss_.item()):
                    losses_.update(loss_.item())
                else:
                    print("NaN in model loss!!")

                # compute gradient and do SGD step
                loss_.backward()
                optimizer.step()

                t_writer_1.add_scalar("lr", optimizer.param_groups[0]['lr'],
                                      global_step=epoch * batch_per_epoch + i)

                if scheduler is not None:
                    scheduler.step()

                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()
                # Display progress
                progress.display(i)

            t_writer_1.add_scalar(f"SummaryLoss/train", losses_.avg, epoch)

            te = time.perf_counter()
            print(f"Train Epoch done in {te - ts} s")
            torch.cuda.empty_cache()

            # Validate at the end of epoch every val step
            if (epoch + 1) % args.val == 0:
                validation_loss_1, validation_dice = step(val_loader, model_1, criterian_val, metric, epoch, t_writer_1,
                                                          save_folder=args.save_folder_1,
                                                          patients_perf=patients_perf)

                t_writer_1.add_scalar(f"SummaryLoss", validation_loss_1, epoch)
                t_writer_1.add_scalar(f"SummaryDice", validation_dice, epoch)

                if validation_dice > best_1:
                    print(f"Saving the model with DSC {validation_dice}")
                    best_1 = validation_dice
                    model_dict = model_1.state_dict()
                    save_checkpoint(
                        dict(
                            epoch=epoch,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                        ),
                        save_folder=args.save_folder_1, )

                ts = time.perf_counter()
                print(f"Val epoch done in {ts - te} s")
                torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("Stopping training loop, doing benchmark")
            break


def connected_domain_2(image):
    Image = image
    image = image[0].cpu().numpy()
    image = image[0, ...]
    cca = sitk.ConnectedComponentImageFilter()
    cca.SetFullyConnected(True)
    _input = sitk.GetImageFromArray(image.astype(np.uint8))
    output_ex = cca.Execute(_input)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(output_ex)
    num_label = cca.GetObjectCount()
    num_list = [i for i in range(1, num_label+1)]
    if num_list == []:
        return Image
    else:
        area_list = []
        for l in range(1, num_label + 1):
            area_list.append(stats.GetNumberOfPixels(l))
        num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
        final_label_list = [num_list_sorted[0]]

        for idx, i in enumerate(num_list_sorted[1:]):
            if area_list[i-1] >= 200:
                final_label_list.append(i)
            else:
                break
        output = sitk.GetArrayFromImage(output_ex)
        for one_label in num_list:
            if one_label in final_label_list:
                continue
            x, y, z, w, h, d = stats.GetBoundingBox(one_label)
            one_mask = (output[z: z + d, y: y + h, x: x + w] != one_label)
            output[z: z + d, y: y + h, x: x + w] *= one_mask
        mask = (output > 0).astype(np.float32)
        mask = torch.from_numpy(mask)
        Image[0][0, ...] = mask
        return Image


def step(data_loader, model, criterion: EDiceLoss_Val, metric, epoch, writer, save_folder=None, patients_perf=None):

    # Setup
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    mode = "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"{mode} Epoch: [{epoch}]")

    end = time.perf_counter()
    metrics = []

    for i, val_data in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        patient_id = val_data["patient_id"]

        model.eval()
        with torch.no_grad():
            val_inputs, val_labels = (
                val_data["image"].cuda(device=NO_GPU),
                val_data["label"].cuda(device=NO_GPU),
            )
            val_outputs = inference(val_inputs, model)
            val_outputs_1 = [post_trans(i) for i in decollate_batch(val_outputs)]
            # if epoch > 100:
            # val_outputs_1 = connected_domain_2(val_outputs_1)

            segs = val_outputs
            targets = val_labels
            loss_ = criterion(segs, targets)
            dice_metric(y_pred=val_outputs_1, y=val_labels)

        if patients_perf is not None:
            patients_perf.append(
                dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item())
            )

        writer.add_scalar(f"Loss/{mode}{''}",
                          loss_.item(),
                          global_step=batch_per_epoch * epoch + i)

        # measure accuracy and record loss_
        if not np.isnan(loss_.item()):
            losses.update(loss_.item())
        else:
            print("NaN in model loss!!")

        metric_ = metric(segs, targets)
        metrics.extend(metric_)

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        # Display progress
        progress.display(i)

    save_metrics(epoch, metrics, writer, epoch, False, save_folder)
    writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)

    dice_values = dice_metric.aggregate().item()
    dice_metric.reset()
    dice_metric_batch.reset()

    return losses.avg, dice_values


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
