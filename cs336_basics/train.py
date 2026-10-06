import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import wandb

from cs336_basics.config import TrainConfig
from cs336_basics.transformer import *


def train_lm(config: TrainConfig, project_name="cs336_1", name="train_lm"):
    save_dir = Path(config.checkpoint.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with (save_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)

    # Connect your training loop here:
    # config.data.train_path
    # config.model.d_model
    # config.optimizer.learning_rate
    # config.training.batch_size

    # expect the train path to be a numpy array
    train_tokens = np.load(
        config.data.train_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    valid_tokens = np.load(
        config.data.valid_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    lm = TransformerLM.from_config(config.model, device=torch.device(config.training.device))
    optimizer = AdamW.from_config(lm.parameters(), config.optimizer)
    init_iter = 0
    if config.checkpoint.resume_from is not None:
        init_iter = load_checkpoint(config.checkpoint.resume_from, lm, optimizer)

    rng = np.random.default_rng(config.training.seed)
    with wandb.init(project=project_name, name=name, config=asdict(config)) as run:
        for step in range(config.training.max_steps - init_iter):
            in_tokens, targets = data_loader(
                train_tokens, config.training.batch_size, config.model.context_length, config.training.device, rng
            )
            logits = lm(in_tokens)
            loss = cross_entropy_softmax(logits, targets)

            lr = get_lr_cosine_schedule(
                t=init_iter + step + 1,
                alpha_max=config.optimizer.lr,
                alpha_min=config.optimizer.min_learning_rate,
                T_w=config.optimizer.warmup_steps,
                T_c=config.training.max_steps,
            )

            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad()
            loss.backward()

            grad_clip(list(lm.parameters()), config.optimizer.grad_clip)

            optimizer.step()
            if step % config.training.log_interval == config.training.log_interval - 1:
                run.log({"train/loss": loss.item(), "train/lr": lr}, step=step)
            if step % config.validation.interval == config.validation.interval - 1:
                with torch.no_grad():
                    val_tokens, val_targets = data_loader(
                        valid_tokens,
                        config.validation.batches,
                        config.model.context_length,
                        config.training.device,
                        rng,
                    )
                    val_logits = lm(val_tokens)
                    val_loss = cross_entropy_softmax(val_logits, val_targets)
                    run.log({"val/loss": val_loss.item()}, step=step)
            if step % config.checkpoint.interval == config.checkpoint.interval - 1:
                save_checkpoint(lm, optimizer, init_iter + step + 1, save_dir / "state.pt")
