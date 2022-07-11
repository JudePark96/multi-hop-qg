import logging
import os
import random
import sys
from functools import partial

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import get_linear_schedule_with_warmup, Seq2SeqTrainer, Trainer
from transformers import (T5Tokenizer, T5ForConditionalGeneration)
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd() + "/../")  # noqa: E402

from src.data_utils.totto.totto_dataset import TottoDataset
from src.desc_generation.config import train_args
from src.common.gpu_utils import init_gpu_params

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def set_seed(seed: int = 13):
  random.seed(seed)
  os.environ['PYHTONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True


def main():
  args = train_args()
  if args.do_train:
    init_gpu_params(args)

    model_name = f"{args.prefix}-seed{args.seed}-bsz{args.per_gpu_train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}-warm{args.warmup_ratio}-{args.model_name.replace('/', '-')}"
    if args.is_master:
      tb_logger = SummaryWriter(os.path.join(args.output_dir.replace("logs", "tflogs")))

      if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(f"output directory {args.output_dir} already exists and is not empty.")
      os.makedirs(args.output_dir, exist_ok=True)
      os.system("touch {}".format(os.path.join(args.output_dir, model_name)))

    set_seed(args.seed)
    if args.n_gpu > 0:
      torch.cuda.manual_seed_all(args.seed)
  logger.info(args)

  if not args.do_train and not args.do_predict:
    raise ValueError(
      "At least one of `do_train` or `do_predict` must be True.")

  # if args.do_predict:
  #   raise ValueError("Currently, only supports `do_train`.")

  tokenizer = T5Tokenizer.from_pretrained(args.model_name)
  model = T5ForConditionalGeneration.from_pretrained(args.model_name)

  tokenizer.add_special_tokens({
    'additional_special_tokens': [
      '<page_title>',
      '</page_title>',
      '<section_title>',
      '</section_title>',
      '<table>',
      '</table>',
      '<cell>',
      '</cell>',
      '<col_header>',
      '</col_header>',
      '<row_header>',
      '</row_header>'
    ]
  })

  model.resize_token_embeddings(len(tokenizer))

  if args.do_train:
    train_dataset = TottoDataset(args.train_file)
    train_sampler = RandomSampler(train_dataset) if not args.multi_gpu else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size,
                                  pin_memory=True, num_workers=args.num_workers,
                                  collate_fn=partial(TottoDataset.collate_fn, tokenizer=tokenizer,
                                                     max_len=args.max_len))
    logger.info(f"Num of train batches: {len(train_dataloader)}")
  elif args.do_predict:
    dev_dataset = TottoDataset(args.predict_file)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.predict_batch_size,
                                pin_memory=True, num_workers=args.num_workers,
                                collate_fn=partial(TottoDataset.collate_fn, tokenizer=tokenizer,
                                                   max_len=args.max_len))
    logger.info(f"Num of dev batches: {len(dev_dataloader)}")

  if args.do_predict:
    model.load_state_dict(torch.load(args.checkpoint_model_path, map_location='cpu'))

  model.zero_grad()
  model.cuda()

  logger.info(model)

  if args.do_train:
    logger.info(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_parameters = [
      {'params': [p for n, p in model.named_parameters() if not any(
        nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
      {'params': [p for n, p in model.named_parameters() if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = Adam(optimizer_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.fp16:
      scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

  # Distributed training (should be after apex fp16 initialization)
  if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device_ids[args.local_rank]],
                                                      output_device=args.device_ids[args.local_rank],
                                                      find_unused_parameters=True)
  if args.do_train:
    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, n_gpu)
    global_step = 0  # gradient update step
    batch_step = 0  # forward batch count
    model.train()

    logger.info("main: ")
    logger.info("length: dataset: {}, dataloader: {}".format(len(train_dataset), len(train_dataloader)))

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    warmup_steps = t_total * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    model.zero_grad()

    logger.info('Start training....')
    for epoch in range(int(args.num_train_epochs)):
      if args.multi_gpu:
        train_sampler.set_epoch(epoch * 1000)

      iter_bar = tqdm(train_dataloader, desc="Iter", disable=not args.is_master)
      iter_loss = 0.

      for step, batch in enumerate(iter_bar):
        batch_step += 1
        batch = {k: v.cuda() for k, v in batch.items()}
        model.train()

        if args.fp16:
          with torch.cuda.amp.autocast(enabled=args.fp16):
            loss = model(**batch).loss
        else:
          loss = model(**batch).loss

        if args.n_gpu > 1:
          loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
          loss = loss / args.gradient_accumulation_steps

        if args.fp16:
          scaler.scale(loss).backward()
          scaler.unscale_(optimizer)
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        else:
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        if (batch_step + 1) % args.gradient_accumulation_steps == 0:
          if args.fp16:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
          else:
            optimizer.step()
            optimizer.zero_grad()
          scheduler.step()
          model.zero_grad()
          global_step += 1

          if args.is_master:
            tb_logger.add_scalar('batch_train_loss', loss.item(), global_step)

        iter_loss += loss.item()

        iter_bar.set_postfix({
          "epoch": f"{epoch}",
          "global_steps": f"{global_step}",
          "learning_rate": f"{scheduler.get_last_lr()[0]:.10f}",
          "mean_loss": f"{iter_loss / (step + 1) * args.gradient_accumulation_steps:.5f}",
          "last_loss": f"{loss.item() * args.gradient_accumulation_steps:.5f}",
        })

      if args.is_master:
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, f"checkpoint_ep{epoch}.pt"))

    logger.info("Training finished!")
  elif args.do_predict:
    generated = []
    with torch.no_grad():
      iter_bar = tqdm(dev_dataloader, desc="Dev Iteration")
      for step, batch in enumerate(iter_bar):
        input_ids = batch['input_ids'].cuda()
        outputs = model.generate(input_ids,
                                 max_length=300,
                                 num_beams=args.num_beams,
                                 early_stopping=True)

        for generation in tokenizer.batch_decode(outputs, skip_special_tokens=True):
          generated.append(generation)

    with open(args.predict_path, 'a', encoding='utf-8') as f:
      for i in generated:
        f.write(i + '\n')

    logger.info(f'{args.predict_path} saved!')


if __name__ == "__main__":
  main()
