"""
distill.py: Knowledge distillation from a large teacher model (e.g. GPT-4o via OpenAI API)
to the smaller student GPT-2 model defined in gpt2.py.

Steps for distillation:
1. Generate teacher outputs:
   - Supply prompts to teacher API (e.g. OpenAI ChatCompletion with model="gpt-4o").
   - Collect teacher-generated completions.
   - Save prompt-completion pairs to a JSONL file.

2. Prepare student model and dataset:
   - Load prompt-completion pairs.
   - Tokenize using tiktoken (GPT-2 encoding).
   - Create PyTorch Dataset that yields input_ids and target_ids.

3. Train student:
   - Initialize student GPT model (gpt2.GPT).
   - Use cross-entropy loss on student outputs vs teacher-completion tokens.
   - Optionally compute loss only over completion region.
   - Optimize with AdamW and learning-rate schedule.

4. Save distilled student checkpoint.

Example usage:
  python distill.py generate --prompts prompts.txt --output teacher_data.jsonl
  python distill.py train --data teacher_data.jsonl --output student.pt
"""

import os
import json
import argparse
from typing import List

import openai
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

import tiktoken
from gpt2 import GPT, GPTConfig


class TeacherDataGenerator:
    """Generate teacher completions via OpenAI API."""
    def __init__(self, model_name: str = "gpt-4o"):
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key, "Set OPENAI_API_KEY environment variable"
        # initialize the OpenAI client (v1+ library)
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self,
                 prompts: List[str],
                 output_path: str,
                 max_tokens: int = 50,
                 n: int = 1,
                 temperature: float = 0.7,
                 batch_size: int = 8):
        """
        Generate n completions per prompt using parallel chat requests to a GPT-4o teacher.
        batch_size controls the parallelism (max concurrent API calls).
        Writes JSONL records: {"prompt": ..., "completion": ...}.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        # helper: call chat API for single prompt
        def fetch(prompt_text):
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt_text}],
                max_tokens=max_tokens,
                n=n,
                temperature=temperature,
            )
            # extract texts from choices
            texts = [choice.message.content for choice in resp.choices]
            return prompt_text, texts

        # perform parallel API calls
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(fetch, p) for p in prompts]
            with open(output_path, 'w', encoding='utf-8') as fout:
                for fut in as_completed(futures):
                    prompt_text, texts = fut.result()
                    for txt in texts:
                        record = {'prompt': prompt_text, 'completion': txt}
                        fout.write(json.dumps(record, ensure_ascii=False) + '\n')


class DistillationDataset(Dataset):
    """PyTorch Dataset for prompt→teacher-completion pairs."""
    def __init__(self, data_path: str, block_size: int = 1024):
        self.block_size = block_size
        self.enc = tiktoken.get_encoding("gpt2")
        self.records = []
        with open(data_path, "r", encoding="utf-8") as fin:
            for line in fin:
                self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        text = rec["prompt"] + rec["completion"]
        tokens = self.enc.encode(text)
        if len(tokens) > self.block_size:
            tokens = tokens[-self.block_size:]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids


def collate_fn(batch):
    """Pad to longest sequence in batch."""
    input_ids, target_ids = zip(*batch)
    max_len = max(len(ids) for ids in input_ids)
    B = len(batch)
    x = torch.full((B, max_len), fill_value=0, dtype=torch.long)
    y = torch.full((B, max_len), fill_value=-100, dtype=torch.long)
    for i, (inp, tgt) in enumerate(zip(input_ids, target_ids)):
        L = inp.size(0)
        x[i, :L] = inp
        y[i, :L] = tgt
    return x, y


def train_student(data_path: str,
                  output_path: str,
                  epochs: int = 3,
                  batch_size: int = 8,
                  lr: float = 5e-5,
                  device: str = "cuda"):
    """Fine-tune the student GPT model on teacher-generated data."""
    dataset = DistillationDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    config = GPTConfig()
    student = GPT(config).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr)

    student.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = student(x, y)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=-100)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, loss: {avg:.4f}")
    torch.save({"model": student.state_dict(), "config": student.config}, output_path)


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Pipeline")
    sub = parser.add_subparsers(dest="cmd")

    gen = sub.add_parser("generate", help="Generate teacher data")
    gen.add_argument("--prompts", type=str, required=True, help="Text file with one prompt per line")
    gen.add_argument("--output", type=str, required=True, help="Output JSONL path")
    gen.add_argument("--model", type=str, default="gpt-4o", help="Teacher model name")
    gen.add_argument("--max_tokens", type=int, default=50,
                     help="maximum tokens to generate per completion")
    gen.add_argument("-n", "--num_samples", type=int, default=1,
                     help="number of completions to generate per prompt")
    gen.add_argument("--temp", "--temperature", type=float, default=0.7,
                     help="sampling temperature for generation")

    td = sub.add_parser("train", help="Train student on generated data")
    td.add_argument("--data", type=str, required=True, help="Teacher data JSONL")
    td.add_argument("--output", type=str, required=True, help="Path to save student checkpoint")
    td.add_argument("--epochs", type=int, default=3)
    td.add_argument("--bs", type=int, default=8, help="Batch size")
    td.add_argument("--lr", type=float, default=5e-5)
    td.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    if args.cmd == "generate":
        # load prompts
        with open(args.prompts, "r", encoding="utf-8") as f:
            prompts = [l.strip() for l in f if l.strip()]
        teacher = TeacherDataGenerator(model_name=args.model)
        teacher.generate(
            prompts,
            args.output,
            max_tokens=args.max_tokens,
            n=args.num_samples,
            temperature=args.temp,
        )
    elif args.cmd == "train":
        train_student(args.data, args.output, epochs=args.epochs, batch_size=args.bs, lr=args.lr, device=args.device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()