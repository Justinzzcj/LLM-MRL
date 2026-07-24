import os
import sys
import random
import argparse
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.LLM_MRL import LLM_MRL
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, forward_model, vali, test
from read_prompt import load_content

warnings.filterwarnings("ignore")

fix_seed = 2026
os.environ["PYTHONHASHSEED"] = str(fix_seed)
random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)
np.random.seed(fix_seed)
torch.backends.cudnn.deterministic = True


def main(args):
    mses = []
    maes = []
    result_file = f"{args.model_id}.txt"
    with open(result_file, "w", encoding="utf-8"):
        pass

    for ii in range(args.itr):
        setting = "{}_sl{}_ll{}_pl{}_dm{}_nh{}_eb{}_itr{}".format(
            args.model_id,
            336,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.embed,
            ii,
        )
        path = os.path.join(args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        if args.freq == 0:
            args.freq = "h"

        train_data, train_loader = data_provider(args, "train")
        vali_data, vali_loader = data_provider(args, "val")
        test_data, test_loader = data_provider(args, "test")
        scaler = train_data.scaler

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if args.model == "LLM_MRL":
            model = LLM_MRL(args, device)
        model.to(device)

        model_optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        if args.loss_func == "mse":
            criterion = nn.MSELoss()
        elif args.loss_func == "mae":
            criterion = nn.L1Loss()
        elif args.loss_func == "smape":
            class SMAPE(nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, pred, true):
                    return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))

            criterion = SMAPE()
        else:
            raise ValueError(f"Unsupported loss function: {args.loss_func}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

        for epoch in range(args.train_epochs):
            train_loss = []
            epoch_time = time.time()

            model.train()
            train_pbar = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Train {epoch + 1}/{args.train_epochs}",
                dynamic_ncols=True,
                ascii=True,
                leave=True,
                file=sys.stdout,
            )
            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_pbar:
                model_optim.zero_grad()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

                outputs = forward_model(model, batch_x, ii, batch_x_mark)
                outputs = outputs[:, -args.pred_len:, :]
                batch_y = batch_y[:, -args.pred_len:, :].to(device)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
                train_pbar.set_postfix(loss=f"{loss.item():.5f}")

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
            print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f}".format(epoch + 1, train_loss, vali_loss))

            if args.cos:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]["lr"]))
            else:
                adjust_learning_rate(model_optim, epoch + 1, args)

            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + "/checkpoint.pth"
        model.load_state_dict(torch.load(best_model_path))
        mse, mae = test(model, test_data, test_loader, args, device, ii, scaler)
        mses.append(mse)
        maes.append(mae)
        with open(result_file, "a", encoding="utf-8") as f:
            f.write("experiment {}: mse = {:.4f}, mae = {:.4f}\n".format(ii + 1, mse, mae))

    mse_summary = "mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses))
    mae_summary = "mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes))
    print(mse_summary)
    print(mae_summary)
    with open(result_file, "a", encoding="utf-8") as f:
        f.write("{}\n{}\n".format(mse_summary, mae_summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLLM")
    parser.add_argument("--model_id", type=str, default="LLM_MRL8_trans2_inchannel3_sl12", help="model id")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

    parser.add_argument("--root_path", type=str, default=r"./dataset/SDWPF")
    parser.add_argument("--data_path", type=str, default="SDWPF.npz")
    parser.add_argument("--dataset_name", type=str, default="SDWPF")
    parser.add_argument("--freq", type=str, default="10min")
    parser.add_argument("--embed", type=str, default="timeF")

    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--label_len", type=int, default=6)
    parser.add_argument("--vars", type=int, default=134)
    parser.add_argument("--prompt_max_len", type=int, default=512, help="maximum prompt characters before tokenization")

    parser.add_argument("--decay_fac", type=float, default=0.75)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_epochs", type=int, default=100, help="train epochs")
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--patience", type=int, default=3)

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--loss_func", type=str, default="mse")
    parser.add_argument("--pretrain", type=int, default=1)
    parser.add_argument("--model", type=str, default="LLM_MRL", help="model name")
    parser.add_argument("--tmax", type=int, default=10)

    parser.add_argument("--itr", type=int, default=3, help="experiments times")
    parser.add_argument("--cos", type=int, default=0)
    args = parser.parse_args()
    args.description = load_content()

    main(args)
