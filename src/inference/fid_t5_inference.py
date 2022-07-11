import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer, BeamSearchScorer, LogitsProcessorList, MinLengthLogitsProcessor, \
  MaxLengthCriteria, StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

sys.path.append(os.getcwd() + "/../")  # noqa: E402

from src.common.misc import read_pkl_binary
from src.data_utils.hotpotqa.feature_util import convert_features_to_dataset
from src.modeling.modeling_fid_t5 import FiDT5

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--save_checkpoints_dir", type=str, default="checkpoints/")
  parser.add_argument("--checkpoint_bin", type=str, default="checkpoints/")
  parser.add_argument("--output_file_name", type=str, default="")
  parser.add_argument("--config_name_or_path", type=str, default='../resources/')
  parser.add_argument("--dev_features_path", type=str, default='')
  parser.add_argument("--test_features_path", type=str, default='')
  parser.add_argument("--batch_size", type=int, default=16)
  parser.add_argument("--num_beams", type=int, default=5)
  parser.add_argument("--max_length", type=int, default=128)
  parser.add_argument("--num_workers", type=int, default=30)
  parser.add_argument("--is_cuda", action='store_true')
  parser.add_argument("--is_early_stopping", action='store_true')

  return parser


def evaluate(args, model: FiDT5, tokenizer: T5Tokenizer, loader: DataLoader, eval_type: str = 'Dev') -> List[
  Dict[str, Any]]:
  generated_output = []

  with torch.no_grad():
    for idx, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Evaluating {eval_type} Dataset ...'):
      encoder_input_ids, encoder_attention_mask, labels, entity_start_end = batch[0].cuda(), \
                                                                            batch[1].cuda(), \
                                                                            batch[3], \
                                                                            batch[4]
      bsz, n_passages, seq_length = encoder_input_ids.shape
      input_ids, attention_mask = encoder_input_ids.view(bsz * n_passages, seq_length), \
                                  encoder_attention_mask.view(bsz * n_passages, seq_length)
      encoder_outputs = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

      # [bs x (n_passages * seq_length)]
      last_hidden_state = encoder_outputs.last_hidden_state

      last_hidden_state = last_hidden_state.view(bsz, n_passages, seq_length, -1)
      last_hidden_state = last_hidden_state.view(bsz, n_passages * seq_length, -1)
      last_hidden_state = last_hidden_state.repeat_interleave(args.num_beams, dim=0)

      encoder_outputs.last_hidden_state = last_hidden_state
      model_kwargs = {"encoder_outputs": encoder_outputs}

      beam_scorer = BeamSearchScorer(batch_size=bsz, num_beams=args.num_beams, do_early_stopping=args.is_early_stopping,
                                     device=model.device)

      dec_input_ids = torch.ones((args.num_beams * bsz, 1), device=model.device,
                                 dtype=torch.long) * model.config.decoder_start_token_id
      output = model.beam_search(dec_input_ids, beam_scorer,
                                 stopping_criteria=StoppingCriteriaList(
                                   [MaxLengthCriteria(max_length=args.max_length)]), **model_kwargs)
      labels = labels.tolist()

      if idx < 10:
        logger.info(f'{"*" * 10} Example {idx} {"*" * 10}')

        logger.info(f'{"*" * 10} Generation Output {"*" * 10}')
        for i in output:
          logger.info(tokenizer.decode(i, skip_special_tokens=True))

        logger.info('***' * 20)

        logger.info(f'{"*" * 10} Target Question {"*" * 10}')

        for i in labels:
          i = i[:i.index(-100)]
          logger.info(tokenizer.decode(i, skip_special_tokens=True))

      input_ids = input_ids.view(bsz, n_passages, seq_length)

      for i in range(bsz):
        encoder_input_ids = input_ids[i, :]
        encoder_decoded = tokenizer.batch_decode(encoder_input_ids)
        encoder_decoded = [e[:e.find('<pad>')].strip() for e in encoder_decoded]
        # ([bsz, n_passages, max_entity_length, start,end)])
        positions = entity_start_end[i]

        decoded_entities = []
        answer = None
        for idx, (inputs, p) in enumerate(zip(encoder_input_ids, positions)):
          # Entity 를 정확하게 불러오기 위해서는 </ANS> 의 위치 Index + 1 을 해줘야 함!!!
          answer_end_token_idx = (inputs == tokenizer.convert_tokens_to_ids(tokenizer.tokenize('</ANS>'))[0]).nonzero().squeeze(dim=-1)[0].detach().cpu().item() + 1
          if idx == 0:
            # 여기서만 answer_end_token_idx - 1 을 함.
            answer = tokenizer.decode(inputs[1:answer_end_token_idx - 1]).strip()
          decoded = [tokenizer.decode(inputs[answer_end_token_idx + _[0]:answer_end_token_idx + _[1]]).strip() for _ in
                     p if _[0] != -100]
          decoded_entities.append(decoded)

        target_question = labels[i]
        target_question = target_question[:target_question.index(-100)]
        target_question = tokenizer.decode(target_question, skip_special_tokens=True)

        generation = tokenizer.decode(output[i], skip_special_tokens=True)
        entry = {
          'inputs': encoder_decoded,
          'n_passages': len(encoder_decoded),
          'entities': decoded_entities,
          'answer': answer,
          'label': target_question,
          'generated_question': generation
        }
        generated_output.append(entry)

  return generated_output


def main():
  args = get_parser().parse_args()
  logger.info(f'{"*" * 20} Configuration {"*" * 20}')
  logger.info(vars(args))

  dev_loader, test_loader = None, None

  if args.dev_features_path != '':
    dev_features = read_pkl_binary(args.dev_features_path)
    dev_dataset = convert_features_to_dataset(dev_features)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

  if args.test_features_path != '':
    test_features = read_pkl_binary(args.test_features_path)
    test_dataset = convert_features_to_dataset(test_features)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

  model = FiDT5.from_pretrained(args.config_name_or_path)
  tokenizer = T5Tokenizer.from_pretrained(args.config_name_or_path)

  model.load_state_dict(torch.load(os.path.join(args.save_checkpoints_dir, args.checkpoint_bin), map_location='cpu'))

  if args.is_cuda:
    model.cuda()

  model.eval()

  if dev_loader is not None:
    output = evaluate(args, model, tokenizer, dev_loader, 'Dev')
    with open(os.path.join(args.save_checkpoints_dir, args.output_file_name), 'w', encoding='utf-8') as f:
      json.dump(output, f, ensure_ascii=False, indent=2)

  if test_loader is not None:
    output = evaluate(args, model, tokenizer, test_loader, 'Test')
    with open(os.path.join(args.save_checkpoints_dir, args.output_file_name), 'w', encoding='utf-8') as f:
      json.dump(output, f, ensure_ascii=False, indent=2)


def debug_main():
  dev_loader, test_loader = None, None

  config_name_or_path = '../../resources/fid-t5-base-add-tokens'
  dev_features_path = '../../resources/hotpotqa/preprocessed/t5_dev_features.pkl.gz'
  save_checkpoints_dir = "../../checkpoints/hotpotqa/FiDT5/E5_B16_WARM0.06_NORM1.0_LR3e-05/"
  checkpoint_bin = "4epoch.pth"
  batch_size, num_workers = 16, 32

  if dev_features_path != '':
    dev_features = read_pkl_binary(dev_features_path)
    dev_dataset = convert_features_to_dataset(dev_features)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, num_workers=num_workers)

  model = FiDT5.from_pretrained(config_name_or_path)
  tokenizer = T5Tokenizer.from_pretrained(config_name_or_path)

  model.load_state_dict(torch.load(os.path.join(save_checkpoints_dir, checkpoint_bin), map_location='cpu'))

  model.cuda()
  model.eval()

  args = get_parser().parse_args()
  args.num_beam = 5
  args.batch_size = 8

  if dev_loader is not None:
    evaluate(args, model, tokenizer, dev_loader, 'Dev')

  if test_loader is not None:
    evaluate(args, model, tokenizer, test_loader, 'Test')


if __name__ == '__main__':
  main()
  # debug_main()
