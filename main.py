import argparse
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle as pkl
from pathlib import Path

from tools import ECPNDataset
from model.ECPN import ECPN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', default="./data/example.fasta")
    parser.add_argument('-o', '--output_dir', default="./result")
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    return parser.parse_args()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = ECPNDataset(args.input_file)
    dataloader = DataLoader(dataset, 32)
    with open('./model/vocab.pkl', 'rb') as f:
        vocab = pkl.load(f)
    vocab_idx2ec = {v: k for k, v in vocab.items()}
    CONFIG = {
        "labels": [7, 70, 212, 2234],
        "alpha": 0.5,
        "checkpoint_path": None,
    }
    device = args.device
    model = ECPN(CONFIG)
    model.load_state_dict(torch.load('./model/weight.pth'))
    model = model.to(device)
    accs = []
    pr = []
    with torch.no_grad():
        model.eval()
        for step, (acc, x) in tqdm(enumerate(dataloader), total=len(dataloader), smoothing=0.9):
            y_pred = model(x.to(device))  # (batch, class)
            pred = y_pred["scores"].cpu().data.numpy()
            pred[pred < args.threshold] = 0
            pred[pred >= args.threshold] = 1
            pr.append(pred)
            accs.append(acc)
    pr = np.concatenate(pr, axis=0)
    accs = np.concatenate(accs, axis=0)
    labels = CONFIG["labels"]
    with open(output_dir.joinpath('result.txt'), 'w') as f:
        f.writelines(f"Query ID\tPredicted EC number\n")
        for idx in range(len(accs)):
            prediction_value = pr[idx]
            index_of_ec = np.where(prediction_value == 1)[0]
            if len(index_of_ec) == 0:
                continue
            for index in index_of_ec:
                if index >= sum(labels) - labels[-1]:
                    f.writelines(f"{accs[idx]}\tEC:{vocab_idx2ec[index]}\n")


if __name__ == "__main__":
    start = time.process_time()
    args = parse_args()
    main(args)
    end = time.process_time()
    print("All time: %s" % (end - start))
