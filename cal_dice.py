import numpy as np

path1 = r'E:\pcx\VT-UNet\runs_twostage\logs_base\model_1\val.txt'#ET:0.802, TC:0.83995, WT:0.88314, max_avg_dice:0.8416966666666666
path2 = r'E:\pcx\VT-UNet\runs_onestage\logs_base\model_1\val.txt'#ET:0.79367, TC:0.83191, WT:0.8846, max_avg_dice:0.8367266666666667
path3 = r'E:\pcx\VT-UNet\runs_TransBTS\logs_base\model_1\val.txt'#ET:0.76287, TC:0.79252, WT:0.86678, max_avg_dice:0.8073899999999999
path4 = r'E:\pcx\VT-UNet\runs_twostage_zsocre\logs_base\model_1\val.txt'#ET:0.80247, TC:0.83505, WT:0.89364, max_avg_dice:0.8437199999999999
path5 = r'E:\pcx\VT-UNet\runs_twostage_wdice\logs_base\model_1\val.txt'#ET:0.79962, TC:0.83886, WT:0.88621, max_avg_dice:0.8415633333333333
path6 = r'E:\pcx\VT-UNet\runs_twostage_zsocre_warmup\logs_base\model_1\val.txt'#ET:0.80926, TC:0.83471, WT:0.89004, max_avg_dice:0.8446699999999999
path7 = r'E:\pcx\CU-Trans\runs\logs_base\model_1\val.txt'

max_ET_dice = 0
max_WT_dice = 0
max_TC_dice = 0
max_avg_dice = 0
max_epoch = 0
for line in open(path7):
        EPOCH =line.split(' :Val')[0]
        EPOCH = np.array(EPOCH.split('Epoch ')[1], dtype=np.float)
        ET = np.array(line.split('ET : ')[1][0:7], dtype=np.float)
        WT = np.array(line.split('WT : ')[1][0:7], dtype=np.float)
        TC = np.array(line.split('TC : ')[1][0:7], dtype=np.float)
        avg_dice = (ET + WT + TC) / 3
        if max_avg_dice < avg_dice:
                max_avg_dice = avg_dice
                max_ET_dice = ET
                max_WT_dice = WT
                max_TC_dice = TC
                max_epoch = EPOCH

print(f"max epoch:{max_epoch}, ET:{max_ET_dice}, TC:{max_TC_dice}, WT:{max_WT_dice}, max_avg_dice:{max_avg_dice}")



