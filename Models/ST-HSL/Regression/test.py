# test.py

from model import *
import torch
from Params import args
from DataHandler import DataHandler
from engine import test
from train import makePrint

def main():
    device = torch.device(args.device)
    handler = DataHandler()
    model = STHSL()
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print('model load successfully')

    with torch.no_grad():
        reses = test(model, handler)
        with open("/content/drive/MyDrive/Thesis/ST-HSL/Results_AIST_CATS_c_12h/Comm_" + str(args.community) + ".txt", "w") as f:
          f.write(str(reses))

    print(makePrint('Best', args.epoch, reses))
if __name__ == "__main__":
    main()