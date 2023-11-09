import argparse
import json
import numpy as np
import torch
import tomli
from os import mkdir
from dataclasses import dataclass
from os.path import isfile, isdir, join
from typing import List, Callable
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
from pytorch_metric_learning.losses import MultiSimilarityLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner
from pytorch_metric_learning.samplers import MPerClassSampler
from src.data_loader import Dataset
from src.bert import BertConfigs
def print_config(config: dict):
    '''Prints configuration parameters'''
    print("[DATA CONFIG]")
    for key, val in config["DATA"].items():
        print(f"{key} = {val}", flush=True)
    print("")


def main(config: dict, cp_path: str, logdir: str, verbose: bool):
    '''Builder training set'''
    # --- SETUP TRAIN --------------------------------------

    # config
    if verbose: print_config(config)
    dc = config["DATA"]


    # load train, val
    dataset = Dataset(dc["rawdatapath"])
      # init model
    model = Bert(BertConfigs())
    if verbose: print(f"device: {model.device}\n---", flush=True)

    # entity linking (EL) loss setup
    if tc["loss"] == "multisimilarityloss":
        loss_func = MultiSimilarityLoss(
            alpha=tc["alpha"],
            beta=tc["beta"],
            base=tc["base"]
        )
        miner = MultiSimilarityMiner(
            epsilon=tc["epsilon"]
        )
    else:  # ontologicalloss
        miner = MultiSimilarityMiner(
            epsilon=tc["epsilon"]
        )
        loss_func = OntologicalTripletLoss(
            ontopath=dc["ontopath"],
            margin=tuple(tc["margin"])
        )

    # optimizer setup
    if tc["optim"] == "Adam":
        optimizer = Adam(
            model.parameters(),
            lr=tc["lr"],
            weight_decay=tc["wdecay"]
        )
    else:
        optimizer = SGD(
            model.parameters(),
            lr=tc["lr"],
            momentum=tc["momentum"]
        )

    # training setup
    sampler = MPerClassSampler(
        labels=dataset.labels(),
        m=tc["m"],
        length_before_new_iter=len(dataset)
    )
    loader = DataLoader(
        dataset,
        batch_size=tc["bsize"],
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
        collate_fn=context_collate_fn
    )
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=tc["end_factor"],
        total_iters=tc["epochs"] * len(loader)
    )
    early_stop = EarlyStop(tc["tolerance"], tc["minlen"])
    if tc["evaluate"]:
        evaluator = RetrievalEvaluator(dc["ontojsonpath"], dc["prefix"])

    # --- TRAIN LOOP ---------------------------------------

    for epoch in range(tc["epochs"]):
        if early_stop.stop: break
        for batchidx, (inputs, el_labels, word_id_labels) in enumerate(loader):
            # train step
            loss = train(
                model, optimizer, scheduler, miner,
                loss_func, inputs, el_labels, word_id_labels
            )

            # validate step
            val_loss = validate(
                model, optimizer, loss_func,
                val_ins, val_el_labs, val_word_id_labs
            )

            # logging to stdout
            if verbose and (batchidx % 2 == 0):
                print(f"e{epoch}b{batchidx}|train:{loss.item():.4f}|", end="")
                print(f"val:{val_loss.item():.4f}", flush=True)

            # early stopping
            if early_stop(val_loss.item()):
                if verbose: print("early stopping.")
                break
            else:
                early_stop.losses.append(val_loss.item())

            # slow training save
            # if batchidx%5==0:
            # save_model_checkpoint(epoch, model, optimizer, loss, cp_path)

        # --- EPOCH evaluation -----------------------------
        model.eval()
        if tc["evaluate"] and ((1 + epoch) % tc["epoch_iter"] == 0):
            if verbose:
                print("---")
                print("computing performance...", flush=True)
            evaluator.load(model, True)

            # EL evaluation
            acc1, acck = evaluator(val_ins, val_el_labs)
            if verbose:
                print(f"e{epoch}|el_acc@1:{acc1}")
                print(f"e{epoch}|el_acc@{tc['k']}:{acck}")
                print("---", flush=True)

            # NER evaluation

            # epoch checkpoint
            save_model_checkpoint(epoch, model, optimizer, loss, cp_path)

    # early stopping checkpoint
    save_model_checkpoint(epoch, model, optimizer, loss, cp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configpath",
        default="configs/train_config.toml",
        help="toml config for model training"
    )
    parser.add_argument(
        "--checkpoint",
        default="resources/checkpoints",
        help="path to dir where model checkpoints are saved"
    )
    parser.add_argument(
        "--logdir",
        default="resources/logdir",
        help="path to dir where training logs are saved"
    )
    parser.add_argument(
        "--verbose",
        default=True
    )
    args = parser.parse_args()
    config = tomli.load(open(args.configpath, "rb"))
    logdir, checkpointdir = verify_args(args, config)
    main(config, checkpointdir, logdir, args.verbose)