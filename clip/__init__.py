from .architecture import *
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .architecture import *
from .loss import *

# globals
device = torch.device("cpu")
config = None
models_dict = {
    "ClipModel": ClipModel,
}


def train_one_epoch(model, optimizer, loss, loader, epoch, device):
    L_totals = []
    temperatue = []
    align = []
    uniform = []
    clip = []
    nce = []
    model.train()

    # loop over all batches
    for step, (batch_ppi_input, batch_img_input, batch_genes) in enumerate(loader):
        if batch_ppi_input.size(0) == 1:
            continue  # Skip batches with only one sample
        latent_ppi, latent_img, logit_scale = model(batch_ppi_input, batch_img_input)
        losses = loss(
            batch_ppi_input, batch_img_input, latent_ppi, latent_img, logit_scale
        )

        total_loss = loss.combine(
            clip_w=config.clip_w,
            nce_w=config.nce_w,
            align_w=config.align_w,
            unif_w=config.unif_w,
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        L_totals.append(total_loss.detach().cpu().numpy())
        clip.append(losses["Clip"].mean().detach().cpu().numpy())
        nce.append(losses["NCE"].mean().detach().cpu().numpy())
        align.append(losses["Alignment"].mean().detach().cpu().numpy())
        uniform.append(losses["Uniform"].mean().detach().cpu().numpy())
        temperatue.append(losses["Temperatue"].data.cpu().numpy())

    return {
        "Loss": np.mean(L_totals),
        "Clip": np.mean(clip),
        "NCE": np.mean(nce),
        "Alignment": np.mean(align),
        "Uniform": np.mean(uniform),
        "Temperatue": np.mean(temperatue),
    }


def clip_fit_predict(
    resultsdir,
    data_ppi,
    data_img,
    config,
    wandb=None,
    test_subset=[],
    index_names=[],
):
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    globals()["device"] = device
    globals()["config"] = config
    # transform inputs to tensor
    transform = ToTensor()
    data_ppi = transform(data_ppi).to(device)
    data_img = transform(data_img).to(device)

    # index names if none input
    if len(index_names) == 0:
        index_names = np.arange(config.n_sample)

    # remove test subset...
    train_subset = np.arange(config.n_sample)
    train_subset = list(set(train_subset) - set(test_subset))
    train_data_ppi = data_ppi[train_subset]
    train_data_img = data_img[train_subset]

    # create model, optimizer, trainloader
    model = models_dict[config.model](
        config.ppidim,
        config.imgdim,
        config.latent_dim,
        config.hidden_dim,
        config.dropout,
    ).to(device)
    loss = MainLoss(config)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    train_loader = DataLoader(
        Protein_Dataset(train_data_ppi, train_data_img),
        batch_size=config.batch_size,
        shuffle=True,
    )
    best_loss = float("inf")

    # Define the custom x axis metric
    # https://docs.wandb.ai/guides/track/log/customize-logging-axes
    if wandb is not None:
        wandb.define_metric("Epoch")
        # Define which metrics to plot against that x-axis
        wandb.define_metric("Loss", step_metric="Epoch")
        wandb.define_metric("Clip", step_metric="Epoch")
        wandb.define_metric("Alignment", step_metric="Epoch")
        wandb.define_metric("Uniform", step_metric="Epoch")
        wandb.define_metric("NCE", step_metric="Epoch")
        wandb.define_metric("Temperatue", step_metric="Epoch")

    # TRAIN WITH Contrastive Loss
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        model.train()
        losses = train_one_epoch(model, optimizer, loss, train_loader, epoch, device)
        elapsed = time.time() - epoch_start_time
        # log every 10 epochs
        if epoch % 10 == 0 and config["verbose"]:
            print("-" * 89)
            print(
                f"""| epoch {1+epoch:3d} | time: {elapsed:5.2f}s | loss {losses["Loss"]:5.2f}"""
            )
            print("-" * 89)

        losses["Epoch"] = epoch
        if wandb is not None:
            wandb.log(losses)

        if losses["Loss"] < best_loss:
            best_loss = losses["Loss"]
            best_model = model.state_dict()

    # SAVE FINAL RESULTS
    model.eval()
    with torch.no_grad():
        latent_ppi, latent_img = model.get_embedding(data_ppi, data_img)

    torch.save(best_model, f"{resultsdir}/model.pth")

    df_ppi = pd.DataFrame(latent_ppi.detach().cpu().numpy(), index=index_names)
    df_img = pd.DataFrame(latent_img.detach().cpu().numpy(), index=index_names)

    avg = (latent_ppi.detach().cpu().numpy() + latent_img.detach().cpu().numpy()) / 2
    df_avg = pd.DataFrame(avg, index=index_names)
    df_concat = pd.concat([df_ppi, df_img], axis=1)

    df_ppi.to_csv(f"{resultsdir}/latent_ppi.txt")
    df_img.to_csv(f"{resultsdir}/latent_img.txt")
    df_avg.to_csv(f"{resultsdir}/avg_latent.txt")
    df_concat.to_csv(f"{resultsdir}/cont_latent.txt")

    return model
