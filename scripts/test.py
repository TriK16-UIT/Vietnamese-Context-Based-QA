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

def test(args):
    set_global_seeds(args.seed)

    print("Testing parameters:")
    print("(" + "; ".join([f'"{arg}": {value}' for arg, value in vars(args).items()]) + ")")
    print("\nStarting testing..\n")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    raw_datasets = load_dataset("utils/viquad.py")

    raw_datasets["test"] = raw_datasets["test"].filter(lambda x: len(x["answers"]["text"]) == 1)

    test_dataset = raw_datasets["test"].map(
        lambda examples: preprocess_validation_dataset(examples, tokenizer=tokenizer, max_length=args.max_length, stride=args.stride),
        batched=True,
        remove_columns=raw_datasets["test"].column_names
    )

    metric = evaluate.load(args.metric)

    test_dataset.set_format("torch")
    test_set = test_dataset.remove_columns(["example_id", "offset_mapping"])
    test_set.set_format("torch")

    test_dataloader = DataLoader(
        test_set,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        pin_memory=True
    )

    device = torch.device(args.device)
    model = AutoModelForQuestionAnswering.from_pretrained(args.pretrained_model)
    model.to(device)

    start_logits = []
    end_logits = []
    print("Evaluation!")
    for batch in tqdm(test_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        start_logits.append(outputs.start_logits.cpu().numpy())
        end_logits.append(outputs.end_logits.cpu().numpy())

    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    start_logits = start_logits[: len(test_dataset)]
    end_logits = end_logits[: len(test_dataset)]

    metrics = compute_metrics(args, metric,
        start_logits, end_logits, test_dataset, raw_datasets["test"]
    )

    print("Test metrics:", metrics)

def parse_args():
    parser = argparse.ArgumentParser(description="Train PhoBERT for Question Answering task")

    parser.add_argument('-metric', type=str, default="squad")
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-checkpoints', type=str, default="checkpoints")
    parser.add_argument('-pretrained_model', type=str, default="vinai/bartpho-syllable-base")

    parser.add_argument('-batch_size', type=int, default=2)

    parser.add_argument('-max_length', type=int, default=1024)
    parser.add_argument('-stride', type=int, default=128)

    parser.add_argument('-n_best', type=int, default=20)
    parser.add_argument('-max_answer_length', type=int, default=200)

    parser.add_argument('-seed', type=int, default=None)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test(args)