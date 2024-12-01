import argparse
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from utils.utils import *
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

def inference(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoints)
    tokenizer.model_input_names.remove("token_type_ids")

    inputs = tokenizer(args.question, args.context, return_tensors="pt")

    model = AutoModelForQuestionAnswering.from_pretrained(args.checkpoints)
    model.to(args.device)
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    print(tokenizer.decode(predict_answer_tokens))

def parse_args():
    parser = argparse.ArgumentParser(description="Perform inference with the trained QA model")
    
    parser.add_argument('-device', type=str, default="cuda")
    parser.add_argument('-checkpoints', type=str, default="checkpoints")
    parser.add_argument('-context', type=str, required=True)
    parser.add_argument('-question', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    inference(args)
