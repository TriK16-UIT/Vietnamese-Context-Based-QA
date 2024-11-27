import argparse
import evaluate
import sys
import numpy as np
import torch
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from datasets import load_dataset
from utils.utils import *
from utils.metric import compute_metrics
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, default_data_collator, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

def train(args):
    set_global_seeds(args.seed)

    print("Training parameters:")
    print("(" + "; ".join([f'"{arg}": {value}' for arg, value in vars(args).items()]) + ")")
    print("\nStarting training...\n")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    #TODO: fix this hardcode
    raw_datasets = load_dataset("utils/viquad.py")
    
    #TODO: check later
    # raw_datasets["train"] = raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) == 1)
    raw_datasets["validation"] = raw_datasets["validation"].filter(lambda x: len(x["answers"]["text"]) == 1)
    # raw_datasets["train"] = raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) == 1)

    train_dataset = raw_datasets["train"].map(
        lambda examples: preprocess_dataset(examples, tokenizer=tokenizer, max_length=args.max_length, stride=args.stride),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    val_dataset = raw_datasets["validation"].map(
        lambda examples: preprocess_validation_dataset(examples, tokenizer=tokenizer, max_length=args.max_length, stride=args.stride),
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )

    metric = evaluate.load(args.metric)

    train_dataset.set_format("torch")
    val_set = val_dataset.remove_columns(["example_id", "offset_mapping"])
    val_set.set_format("torch")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=4
    )

    eval_dataloader = DataLoader(
        val_set,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        pin_memory=True
    )

    device = torch.device(args.device)
    model = AutoModelForQuestionAnswering.from_pretrained(args.pretrained_model)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = args.epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.scheduler,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))

    prev_metrics = None
    for epoch in range(args.epochs):
        # Training
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            epoch_loss += loss.item()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch} - Training Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        start_logits = []
        end_logits = []
        print("Evaluation!")
        for batch in tqdm(eval_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            start_logits.append(outputs.start_logits.cpu().numpy())
            end_logits.append(outputs.end_logits.cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(val_dataset)]
        end_logits = end_logits[: len(val_dataset)]

        metrics = compute_metrics(args, metric,
            start_logits, end_logits, val_dataset, raw_datasets["validation"]
        )
        print(f"epoch {epoch}:", metrics)

        if prev_metrics is None or metrics['f1'] > prev_metrics['f1']:
            print(f"Saving model to {args.output_dir}...")
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)  # Save tokenizer as well
            print("Finished.")
            prev_metrics = metrics



def parse_args():
    parser = argparse.ArgumentParser(description="Train PhoBERT for Question Answering task")

    parser.add_argument('-metric', type=str, default="squad")
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-output_dir', type=str, default="checkpoints")
    parser.add_argument('-scheduler', type=str, default="linear")
    parser.add_argument('-pretrained_model', type=str, default="vinai/bartpho-syllable-base")

    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-lr', type=float, default=2e-5)

    parser.add_argument('-max_length', type=int, default=1024)
    parser.add_argument('-stride', type=int, default=128)

    parser.add_argument('-n_best', type=int, default=20)
    parser.add_argument('-max_answer_length', type=int, default=200)

    parser.add_argument('-seed', type=int, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)