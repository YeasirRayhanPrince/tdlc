### Classification
# train.py

# train.py 

import torch
import numpy as np
import time
from engine import trainer
from Params import args
from utils import seed_torch, makePrint

def main():
    seed_torch()
    device = torch.device('cuda:0')
    engine = trainer(device)
    print("start training...", flush=True)
    train_time = []
    bestRes = None
    eval_bestRes = dict()
    eval_bestRes['macroF1_sklearn'], eval_bestRes['microF1_sklearn'] = 1e6, 1e6
    update = False

    for i in range(1, args.epoch+1):
        t1 = time.time()
        metrics, metrics1 = engine.train()
        print(f'Epoch {i:2d} Training Time {time.time() - t1:.3f}s')
        ret = 'Epoch %d/%d, %s %.4f,  %s %.4f' % (i, args.epoch, 'Train Loss = ', metrics, 'preLoss = ', metrics1)
        print(ret)

        test = (i % args.tstEpoch == 0)
        if test:
            res_eval = engine.eval(True, True)
            val_metrics = res_eval['macroF1_sklearn'] + res_eval['microF1_sklearn']
            val_best_metrics = eval_bestRes['macroF1_sklearn'] + eval_bestRes['microF1_sklearn']
            if (val_metrics) < (val_best_metrics):
                print('%s %.4f, %s %.4f' % ('Val metrics decrease from', val_best_metrics, 'to', val_metrics))
                eval_bestRes['macroF1_sklearn'] = res_eval['macroF1_sklearn']
                eval_bestRes['microF1_sklearn'] = res_eval['microF1_sklearn']
                update = True
            reses = engine.eval(False, True)
            # torch.save(engine.model.state_dict(),
            #            args.save + args.data + "/" + "_epoch_" + str(i) + "_MAE_" + str(round(reses['microF1_sklearn'], 2)) + "_MAPE_" + str(
            #                round(reses['MAPE'], 2)) + ".pth")
            if update:
                print(makePrint('Test', i, reses))
                bestRes = reses
                update = False
        print()
        t2 = time.time()
        train_time.append(t2-t1)
    
    ### BY ALIF
    with open("/content/drive/MyDrive/Thesis/ST-HSL/Results_AIST_CATS_c/Comm_" + str(args.community) + ".txt", "w") as f:
      f.write(str(bestRes))

    print(makePrint('Best', args.epoch, bestRes))

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))