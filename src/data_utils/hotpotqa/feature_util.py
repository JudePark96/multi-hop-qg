import argparse
import json
import logging
import os
import sys
from copy import copy
from functools import partial
from multiprocessing import Pool
from typing import List, Dict, Any, Tuple, Optional

import nltk
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from transformers import BartTokenizer, T5Tokenizer, AutoTokenizer

sys.path.append(os.getcwd() + "/../")  # noqa: E402

from src.common.misc import save_pkl_binary, intersection, read_pkl_binary
from src.data_utils.hotpotqa.dto_wrapper import InputExample, InputFeature

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# PUNCT_WORDS = {'?', ',', '~', '@', ']', '"', '!', '*', '.', '[', '<', '|', '-', '(', '{', '\\', '^', '#', '`', '_', '$',
#                ')', ';', '+', '&', '>', '=', '/', '%', ':', '}'}
PUNCT_WORDS = {"?", ",", ".", "(", ")", "{", "}", "#", "_", "^", "\\", "!", ":", ";"}


def get_parser():
  parser = argparse.ArgumentParser()
  # Common
  parser.add_argument('--preprocessing_type', type=str)  # [input_example or input_feature]
  parser.add_argument('--model_name_or_path', type=str, default='facebook/bart-base')
  parser.add_argument('--max_seq_length', type=int, default=384)
  parser.add_argument('--output_path', required=True, type=str)
  parser.add_argument('--model_type', type=str, default='bart')
  parser.add_argument('--n_threads', type=int, default=20)

  # Creating InputExamples
  parser.add_argument('--hotpotqa_path', type=str)
  parser.add_argument('--entity_path', type=str)
  parser.add_argument('--paragraph_path', type=str)

  # Creating InputFeatures
  parser.add_argument('--example_path', type=str)
  parser.add_argument('--max_query_length', type=int, default=128)
  parser.add_argument('--max_answer_length', type=int, default=64)
  parser.add_argument('--max_entity_length', type=int, default=20)
  parser.add_argument('--max_coref_length', type=int, default=10)

  args = parser.parse_args()

  assert args.model_type in args.model_name_or_path

  return args


def word_tokenize(tokens):
  return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


def convert_idx(text, tokens):
  current = 0
  spans = []
  for token in tokens:
    current = text.find(token, current)
    if current < 0:
      print("Token {} cannot be found".format(token))
      raise Exception()
    spans.append((current, current + len(token)))
    current += len(token)
  return spans


def clean_entity(entity: Tuple[str, str]):
  Type = entity[1]
  Text = entity[0]
  # if Type == "DATE" and ',' in Text:
  #     Text = Text.replace(' ,', ',')
  if '?' in Text:
    Text = Text.split('?')[0]
  # Text = Text.replace("\'\'", "\"")
  # Text = Text.replace("# ", "#")
  Text = Text.replace("''", '" ').replace("``", '" ').lower()
  return Text, Type


def clean_entity_type(entity_type: str) -> str:
  return entity_type.split('-')[-1].strip() if '-' in entity_type else entity_type


def convert_sequence_to_token_with_index(
  tokenizer: AutoTokenizer,
  model_type: str,
  sequence: str,
  max_seq_length: int = 384
) -> Dict[str, Any]:
  cand_indices, tokens_word = [], []
  tokens: List[str] = tokenizer.tokenize(sequence)[:max_seq_length]

  # else case is for T5.
  special_symbol = "Ġ" if 'bart' in model_type or 'roberta' in model_type else "▁"

  for (i, token) in enumerate(tokens):
    if len(cand_indices) >= 1 and not token.startswith(special_symbol) and not (token in PUNCT_WORDS):
      cand_indices[-1].append(i)
      tokens_word[-1] = (tokens_word[-1] + token)
    else:
      cand_indices.append([i])
      tokens_word.append(token.replace(special_symbol, '').lower())

  return {
    'tokens': tokens,
    'tokens_by_word': tokens_word,
    'cand_indices': cand_indices
  }


def remove_duplicated_spans(
  entity_spans: List[Tuple[int, int]],
  seq_length: int
) -> List[Tuple[int, int]]:
  entity_spans.sort(key=lambda x: abs(x[0] - x[1]), reverse=True)

  seq = np.array([0] * seq_length)
  reannotated_entity_spans = []

  for idx, span in enumerate(entity_spans):
    if seq[span[0]] == 0:
      seq[span[0]:span[1]] = idx + 1
      reannotated_entity_spans.append(span)
    else:
      continue

  reannotated_entity_spans.sort(key=lambda x: x[0])
  return reannotated_entity_spans


def read_hotpotqa_fid_examples(
  tokenizer: BartTokenizer,
  hotpotqa_path: str,
  paragraph_path: str,
  entity_path: str,
  model_type: str = 'bart',
  max_seq_length: int = 384
) -> List[InputExample]:
  with open(hotpotqa_path, 'rb') as hf, open(entity_path, 'rb') as ef, open(paragraph_path, 'rb') as pf:
    hotpotqa, entity_data, paragraphs = json.load(hf), json.load(ef), json.load(pf)

  no_answer_count = 0
  input_examples = []

  for (h_idx, case) in tqdm(enumerate(hotpotqa), total=len(hotpotqa)):
    question_type = case['type']
    if question_type == 'comparison':
      continue

    key = case['_id']
    qas_type = case['type']
    orig_answer_text = case['answer']
    orig_question_text = case['question']
    # print(orig_question_text)
    # print(orig_answer_text)
    # print(entity_data[key])

    answer_text_lower = orig_answer_text.replace("''", '" ').replace("``", '" ').lower()
    question_text_lower = orig_question_text.replace("''", '" ').replace("``", '" ').lower()

    answer_tokens = tokenizer.tokenize(' '.join(word_tokenize(answer_text_lower)))
    question_tokens = tokenizer.tokenize(' '.join(word_tokenize(question_text_lower)))
    paragraph_doc_tokens, paragraph_answer_exists_check = [], []
    entity_start_end_positions = []

    for paragraph in paragraphs[key]:
      title, sents = paragraph[0], paragraph[1]
      entities = entity_data[key][title] if title in entity_data[key] else []

      sent = ' '.join(sents)
      sent = sent.strip()
      sent = sent.replace("''", '" ').replace("``", '" ').lower()

      sent_tokens = word_tokenize(sent)
      paragraph_answer_exists_check.append(' '.join(sent_tokens))

      conversion_dict = convert_sequence_to_token_with_index(tokenizer, model_type, ' '.join(sent_tokens),
                                                             max_seq_length=max_seq_length)
      para_tokens, para_tokens_by_word, para_cand_indices = conversion_dict['tokens'], \
                                                            conversion_dict['tokens_by_word'], \
                                                            conversion_dict['cand_indices']
      entity_start_end_position = []

      for entity in entities:
        entity_text, entity_type = clean_entity(entity)
        entity_type = clean_entity_type(entity_type)
        ent_tmp = ' '.join(word_tokenize(entity_text))
        if ent_tmp in para_tokens_by_word:
          entity_index = para_tokens_by_word.index(ent_tmp)
          entity_span = (
            para_cand_indices[entity_index][0],
            para_cand_indices[entity_index][0] + len(para_cand_indices[entity_index]),
            entity_type
          )
        elif ent_tmp in ' '.join(para_tokens_by_word):
          ent_tmp_split = ent_tmp.split(' ')
          all_occur_indices = [i for i, item in enumerate(para_tokens_by_word) if item == ent_tmp_split[0]]
          flag, start_point, end_point = False, -1, -1

          for o in all_occur_indices:
            if ' '.join(para_tokens_by_word[o:o + len(ent_tmp_split)]) == ent_tmp:
              start_point, end_point = o, o + len(ent_tmp_split)
              flag = True

          if flag:
            entity_span_tmp = []
            for i in range(start_point, end_point):
              entity_span_tmp.append((para_cand_indices[i][0], len(para_cand_indices[i])))

            entity_span = (
              entity_span_tmp[0][0],
              entity_span_tmp[0][0] + sum([i[1] for i in entity_span_tmp]),
              entity_type
            )
          else:
            continue
        else:
          continue

        entity_start_end_position.append(entity_span)

      entity_start_end_position = remove_duplicated_spans(entity_start_end_position, seq_length=len(para_tokens))
      entity_start_end_positions.append(entity_start_end_position)
      paragraph_doc_tokens.append(para_tokens)

    # print(len(paragraph_doc_tokens))
    # print(paragraph_doc_tokens)
    # print('***' * 10)
    # print(len(entity_start_end_positions))
    # print(entity_start_end_positions)
    # print('***' * 10)

    decoded_entities = []

    for tokens, positions in zip(paragraph_doc_tokens, entity_start_end_positions):
      decoded_entities.append(
        [(tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens[p[0]:p[1]])), p[2]) for p in positions])

    assert len(decoded_entities) == 2, '%d v.s. %d' % (len(decoded_entities), 2)

    coref_positions = []

    for s_idx, source in enumerate(decoded_entities[0]):
      for t_idx, target in enumerate(decoded_entities[1]):
        is_intersection = intersection(source[0], target[0])
        if is_intersection:
          # If same entity type.
          if source[1] == target[1]:
            coref_positions.append((s_idx, t_idx))

    removed_entity_type_start_end_positions = []

    for arr in entity_start_end_positions:
      arr = [(e[0], e[1]) for e in arr]
      removed_entity_type_start_end_positions.append(arr)

    answer_exists_flag = False

    # check the existence of answer in paragraph.
    for sent in paragraph_answer_exists_check:
      if ' '.join(word_tokenize(answer_text_lower)) in sent:
        answer_exists_flag = True
        break

    if not answer_exists_flag:
      no_answer_count += 1
      continue

    input_examples.append(InputExample(
      qid=key,
      qas_type=qas_type,
      question_tokens=question_tokens,
      answer_tokens=answer_tokens,
      paragraph_tokens=paragraph_doc_tokens,
      coref_positions=coref_positions,
      entity_start_end_positions=removed_entity_type_start_end_positions
    ))

  logger.info(f'counting the case which has no answer: {no_answer_count}')

  return input_examples


def read_hotpotqa_normal_examples(
  tokenizer: Optional[T5Tokenizer],
  hotpotqa_path: str,
  paragraph_path: str,
  entity_path: str,
  model_type: str = 'bart',
  max_seq_length: int = 384
) -> List[InputExample]:
  with open(hotpotqa_path, 'rb') as hf, open(entity_path, 'rb') as ef, open(paragraph_path, 'rb') as pf:
    hotpotqa, entity_data, paragraphs = json.load(hf), json.load(ef), json.load(pf)

  no_answer_count = 0
  input_examples = []

  for (h_idx, case) in tqdm(enumerate(hotpotqa), total=len(hotpotqa)):
    question_type = case['type']
    if question_type == 'comparison':
      continue

    key = case['_id']
    qas_type = case['type']
    orig_answer_text = case['answer']
    orig_question_text = case['question']
    # print(orig_question_text)
    # print(orig_answer_text)
    # print(entity_data[key])

    answer_text_lower = orig_answer_text.replace("''", '" ').replace("``", '" ').lower()
    question_text_lower = orig_question_text.replace("''", '" ').replace("``", '" ').lower()

    answer_tokens = tokenizer.tokenize(' '.join(word_tokenize(answer_text_lower)))
    question_tokens = tokenizer.tokenize(' '.join(word_tokenize(question_text_lower)))
    paragraph_doc_tokens, paragraph_answer_exists_check = [], []
    entity_start_end_positions = []

    for paragraph in paragraphs[key]:
      title, sents = paragraph[0], paragraph[1]
      entities = entity_data[key][title] if title in entity_data[key] else []

      sent = ' '.join(sents)
      sent = sent.strip()
      sent = sent.replace("''", '" ').replace("``", '" ').lower()

      sent_tokens = word_tokenize(sent)
      paragraph_answer_exists_check.append(' '.join(sent_tokens))

      conversion_dict = convert_sequence_to_token_with_index(tokenizer, model_type, ' '.join(sent_tokens),
                                                             max_seq_length=max_seq_length // len(paragraphs[key]))
      para_tokens, para_tokens_by_word, para_cand_indices = conversion_dict['tokens'], \
                                                            conversion_dict['tokens_by_word'], \
                                                            conversion_dict['cand_indices']
      entity_start_end_position = []

      for entity in entities:
        entity_text, entity_type = clean_entity(entity)
        entity_type = clean_entity_type(entity_type)
        ent_tmp = ' '.join(word_tokenize(entity_text))
        if ent_tmp in para_tokens_by_word:
          entity_index = para_tokens_by_word.index(ent_tmp)
          entity_span = (
            para_cand_indices[entity_index][0],
            para_cand_indices[entity_index][0] + len(para_cand_indices[entity_index]),
            entity_type
          )
        elif ent_tmp in ' '.join(para_tokens_by_word):
          ent_tmp_split = ent_tmp.split(' ')
          all_occur_indices = [i for i, item in enumerate(para_tokens_by_word) if item == ent_tmp_split[0]]
          flag, start_point, end_point = False, -1, -1

          for o in all_occur_indices:
            if ' '.join(para_tokens_by_word[o:o + len(ent_tmp_split)]) == ent_tmp:
              start_point, end_point = o, o + len(ent_tmp_split)
              flag = True

          if flag:
            entity_span_tmp = []
            for i in range(start_point, end_point):
              entity_span_tmp.append((para_cand_indices[i][0], len(para_cand_indices[i])))

            entity_span = (
              entity_span_tmp[0][0],
              entity_span_tmp[0][0] + sum([i[1] for i in entity_span_tmp]),
              entity_type
            )
          else:
            continue
        else:
          continue

        entity_start_end_position.append(entity_span)

      entity_start_end_position = remove_duplicated_spans(entity_start_end_position, seq_length=len(para_tokens))
      entity_start_end_positions.append(entity_start_end_position)
      paragraph_doc_tokens.append(para_tokens)

    # print(len(paragraph_doc_tokens))
    # print(paragraph_doc_tokens)
    # print('***' * 10)
    # print(len(entity_start_end_positions))
    # print(entity_start_end_positions)
    # print('***' * 10)

    aggregated_paragraph_doc_tokens, aggregated_entity_start_end_positions = [], []

    assert len(paragraph_doc_tokens) == len(entity_start_end_positions), '%d v.s. %d' % (len(paragraph_doc_tokens),
                                                                                         len(
                                                                                           entity_start_end_positions))

    for p, e in zip(paragraph_doc_tokens, entity_start_end_positions):
      if len(aggregated_paragraph_doc_tokens) == 0:
        aggregated_paragraph_doc_tokens.extend(p)
        aggregated_entity_start_end_positions.extend(e)
      else:
        prev_seq_len = len(aggregated_paragraph_doc_tokens)
        aggregated_paragraph_doc_tokens.extend(p)
        updated_entity_positions = [(prev_seq_len + _[0], prev_seq_len + _[1], _[2]) for _ in e]
        aggregated_entity_start_end_positions.extend(updated_entity_positions)

    # conversion 할 때 maxlength // len(paragraphs[key]) 해줌.
    aggregated_paragraph_doc_tokens = aggregated_paragraph_doc_tokens[:max_seq_length]
    decoded_entities = []
    for p in aggregated_entity_start_end_positions:
      if p[0] > max_seq_length - 1 or p[1] > max_seq_length - 1:
        continue

      entity = tokenizer.decode(tokenizer.convert_tokens_to_ids(aggregated_paragraph_doc_tokens[p[0]:p[1]]))

      if len(entity) == 0:
        continue
      elif tokenizer.unk_token in entity:
        continue
      elif '.' == entity:
        continue
      elif '.' == entity[0]:
        continue

      decoded_entities.append((entity, p[2]))

    coref_positions = []

    for e_idx in range(len(decoded_entities)):
      for i_idx in range(len(decoded_entities)):
        if e_idx == i_idx:
          continue

        s, t = decoded_entities[e_idx], decoded_entities[i_idx]
        is_intersection = intersection(s[0], t[0])
        if is_intersection and s[1] == t[1]:
          if ((i_idx, e_idx) not in coref_positions) and ((e_idx, i_idx) not in coref_positions):
            coref_positions.append((e_idx, i_idx))

    removed_entity_type_start_end_positions = []

    for arr in aggregated_entity_start_end_positions:
      removed_entity_type_start_end_positions.append((arr[0], arr[1]))

    answer_exists_flag = False

    # check the existence of answer in paragraph.
    for sent in paragraph_answer_exists_check:
      if ' '.join(word_tokenize(answer_text_lower)) in sent:
        answer_exists_flag = True
        break

    if not answer_exists_flag:
      no_answer_count += 1
      continue

    input_examples.append(InputExample(
      qid=key,
      qas_type=qas_type,
      question_tokens=question_tokens,
      answer_tokens=answer_tokens,
      paragraph_tokens=aggregated_paragraph_doc_tokens,
      coref_positions=coref_positions,
      entity_start_end_positions=removed_entity_type_start_end_positions
    ))

  logger.info(f'counting the case which has no answer: {no_answer_count}')

  return input_examples


def convert_examples_to_fid_features(
  examples: List[InputExample],
  tokenizer: BartTokenizer,
  max_seq_length: int,
  max_query_length: int,
  max_answer_length: int,
  max_entity_length: int = 20,
  max_coref_length: int = 10,
) -> List[InputFeature]:
  features = []
  answer_denote_tokens = ['<ANS>', '</ANS>']

  for (example_index, example) in enumerate(tqdm(examples)):
    question_tokens = example.question_tokens
    answer_tokens = example.answer_tokens
    paragraph_tokens = example.paragraph_tokens
    entity_start_end_positions = example.entity_start_end_positions
    coref_positions = example.coref_positions

    for e in range(len(entity_start_end_positions)):
      entity_start_end_positions[e] = entity_start_end_positions[e][:max_entity_length]

      while len(entity_start_end_positions[e]) < max_entity_length:
        entity_start_end_positions[e].append([-100, -100])

      assert len(entity_start_end_positions[e]) == max_entity_length, '%d v.s. %d' % (
        len(entity_start_end_positions[e]), max_entity_length)

    coref_positions = coref_positions[:max_coref_length]

    while len(coref_positions) < max_coref_length:
      coref_positions.append([-100, -100])

    answer_tokens = [answer_denote_tokens[0]] + answer_tokens + [answer_denote_tokens[1]]
    input_tokens = [
      tokenizer.convert_tokens_to_ids(copy(answer_tokens) + paragraph)[:max_seq_length + max_answer_length + 2]
      for paragraph in paragraph_tokens]
    attention_mask = []

    for i in input_tokens:
      mask = [1] * len(i)
      while len(i) < max_seq_length + max_answer_length + 2:
        i.append(tokenizer.pad_token_id)
        mask.append(0)

      assert len(i) == max_seq_length + max_answer_length + 2, '%d v.s. %d' % (
        len(i), max_seq_length + max_answer_length + 2)
      assert len(mask) == max_seq_length + max_answer_length + 2, '%d v.s. %d' % (
        len(mask), max_seq_length + max_answer_length + 2)
      assert len(i) == len(mask), '%d v.s. %d' % (len(i), len(mask))

      attention_mask.append(mask)

    label_ids = tokenizer.convert_tokens_to_ids(question_tokens)[:max_query_length - 1] + [tokenizer.eos_token_id]

    if example_index < 10:
      logger.info(f'****************** Example {example_index} ******************')
      for input_idx, tokens in enumerate(input_tokens):
        decoded = tokenizer.decode(tokens).split(tokenizer.pad_token)[0].strip()
        logger.info(f'Passage {input_idx}: {decoded}')

      for p_idx, (paragraph, entity_position) in enumerate(zip(paragraph_tokens, entity_start_end_positions)):
        for p in entity_position:
          if p[0] == -100:
            break

          logger.info(
            f'Passage {p_idx} Entity: {tokenizer.decode(tokenizer.convert_tokens_to_ids(paragraph[p[0]:p[1]])).strip()}')

      logger.info(f'Target Question: {tokenizer.decode(label_ids, skip_special_tokens=True)}')

    decoder_input_ids = [tokenizer.eos_token_id] + label_ids[:-1]

    while len(label_ids) < max_query_length:
      label_ids.append(-100)

    while len(decoder_input_ids) < max_query_length:
      decoder_input_ids.append(tokenizer.pad_token_id)

    assert len(label_ids) == len(decoder_input_ids), '%d v.s. %d' % (len(label_ids), len(decoder_input_ids))

    features.append(InputFeature(
      qid=example.qid,
      input_ids=input_tokens,
      decoder_input_ids=decoder_input_ids,
      attention_mask=attention_mask,
      labels=label_ids,
      entity_start_end_position=entity_start_end_positions,
      coref_position=coref_positions
    ))

  return features


def convert_examples_to_normal_features(
  examples: List[InputExample],
  tokenizer: BartTokenizer,
  max_seq_length: int,
  max_query_length: int,
  max_answer_length: int,
  max_entity_length: int = 20,
  max_coref_length: int = 10,
) -> List[InputFeature]:
  features = []
  answer_denote_tokens = ['<ANS>', '</ANS>']

  for (example_index, example) in enumerate(tqdm(examples)):
    question_tokens = example.question_tokens
    answer_tokens = example.answer_tokens
    paragraph_tokens = example.paragraph_tokens
    entity_start_end_positions = example.entity_start_end_positions
    coref_positions = example.coref_positions

    entity_start_end_positions = entity_start_end_positions[:max_entity_length]
    while len(entity_start_end_positions) < max_entity_length:
      entity_start_end_positions.append([-100, -100])

    coref_positions = coref_positions[:max_coref_length]
    while len(coref_positions) < max_coref_length:
      coref_positions.append([-100, -100])

    answer_tokens = [answer_denote_tokens[0]] + answer_tokens + [answer_denote_tokens[1]]
    input_tokens = tokenizer.convert_tokens_to_ids(copy(answer_tokens)
                                                   + paragraph_tokens)[:max_seq_length + max_answer_length + 2]
    attention_mask = [1] * len(input_tokens)

    while len(input_tokens) < max_seq_length + max_answer_length + 2:
      input_tokens.append(tokenizer.pad_token_id)
      attention_mask.append(0)

    assert len(input_tokens) == max_seq_length + max_answer_length + 2, '%d v.s. %d' % (
        len(input_tokens), max_seq_length + max_answer_length + 2)
    assert len(attention_mask) == max_seq_length + max_answer_length + 2, '%d v.s. %d' % (
        len(attention_mask), max_seq_length + max_answer_length + 2)
    assert len(input_tokens) == len(attention_mask), '%d v.s. %d' % (len(input_tokens), len(attention_mask))

    label_ids = tokenizer.convert_tokens_to_ids(question_tokens)[:max_query_length - 1] + [tokenizer.eos_token_id]

    if example_index < 10:
      logger.info(f'****************** Example {example_index} ******************')
      logger.info(f'Input Tokens: {tokenizer.decode(input_tokens, skip_special_tokens=True)}')

      for p_idx, p in enumerate(entity_start_end_positions):
        if p[0] == -100:
          continue

        logger.info(
          f'Entities {p_idx}: {tokenizer.decode(tokenizer.convert_tokens_to_ids(paragraph_tokens[p[0]:p[1]]))}')

      logger.info(f'Target Question: {tokenizer.decode(label_ids, skip_special_tokens=True)}')

    decoder_input_ids = [tokenizer.eos_token_id] + label_ids[:-1]

    while len(label_ids) < max_query_length:
      label_ids.append(-100)

    while len(decoder_input_ids) < max_query_length:
      decoder_input_ids.append(tokenizer.pad_token_id)

    assert len(label_ids) == len(decoder_input_ids), '%d v.s. %d' % (len(label_ids), len(decoder_input_ids))

    features.append(InputFeature(
      qid=example.qid,
      input_ids=input_tokens,
      decoder_input_ids=decoder_input_ids,
      attention_mask=attention_mask,
      labels=label_ids,
      entity_start_end_position=entity_start_end_positions,
      coref_position=coref_positions
    ))

  return features


def convert_features_to_dataset(input_features: List[InputFeature]) -> TensorDataset:
  input_ids = torch.tensor([i.input_ids for i in input_features])
  attention_mask = torch.tensor([i.attention_mask for i in input_features])
  decoder_input_ids = torch.tensor([i.decoder_input_ids for i in input_features])
  labels = torch.tensor([i.labels for i in input_features])
  entity_start_end = torch.tensor([i.entity_start_end_position for i in input_features])
  coref = torch.tensor([i.coref_position for i in input_features])
  dataset = TensorDataset(input_ids, attention_mask, decoder_input_ids, labels, entity_start_end, coref)
  return dataset


if __name__ == '__main__':
  args = get_parser()
  preprocessing_type = args.preprocessing_type
  # hotpotqa_path, paragraph_path, entity_path = '../../../resources/hotpotqa/hotpot_train_v1.1.json', \
  #                                              '../../../resources/hotpotqa/train/selected_paras_QG.json', \
  #                                              '../../../resources/hotpotqa/train/entities.json'

  if args.model_type == 't5':
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
  elif args.model_type == 'bart':
    tokenizer = BartTokenizer.from_pretrained(args.model_name_or_path)

  if preprocessing_type == 'fid_input_example':
    input_examples = read_hotpotqa_fid_examples(tokenizer,
                                                args.hotpotqa_path,
                                                args.paragraph_path,
                                                args.entity_path,
                                                model_type=args.model_type,
                                                max_seq_length=args.max_seq_length)

    save_pkl_binary(args.output_path, input_examples)
  elif preprocessing_type == 'normal_input_example':
    input_examples = read_hotpotqa_normal_examples(tokenizer,
                                                   args.hotpotqa_path,
                                                   args.paragraph_path,
                                                   args.entity_path,
                                                   model_type=args.model_type,
                                                   max_seq_length=args.max_seq_length)
    save_pkl_binary(args.output_path, input_examples)
  elif preprocessing_type == 'input_feature':
    input_examples = read_pkl_binary(args.example_path)
    input_features = convert_examples_to_fid_features(input_examples, tokenizer,
                                                      args.max_seq_length, args.max_query_length,
                                                      args.max_answer_length, args.max_entity_length,
                                                      args.max_coref_length)

    save_pkl_binary(args.output_path, input_features)
  elif preprocessing_type == 'normal_input_feature':
    input_examples = read_pkl_binary(args.example_path)
    input_features = convert_examples_to_normal_features(input_examples, tokenizer,
                                                         args.max_seq_length, args.max_query_length,
                                                         args.max_answer_length, args.max_entity_length,
                                                         args.max_coref_length)

    save_pkl_binary(args.output_path, input_features)

    # input_ids = torch.tensor([i.input_ids for i in input_features])
    # attention_mask = torch.tensor([i.attention_mask for i in input_features])
    # decoder_input_ids = torch.tensor([i.decoder_input_ids for i in input_features])
    # labels = torch.tensor([i.labels for i in input_features])
    # entity_start_end = torch.tensor([i.entity_start_end_position for i in input_features])
    # coref = torch.tensor([i.coref_position for i in input_features])
    # # print(coref.shape)
    #
    # dataset = TensorDataset(input_ids, decoder_input_ids, attention_mask, labels, entity_start_end, coref)
    # loader = DataLoader(dataset, batch_size=32)
    #
    # for batch in loader:
    #   for b in batch:
    #     print(b.shape)
    #   exit()
    # pass

  # with Pool(args.n_threads) as p:
  #   func_ = partial(read_hotpotqa_examples, tokenizer=tokenizer, hotpotqa_path=hotpotqa_path,
  #                   entity_path=entity_path, model_type=args.model_type, max_seq_length=args.max_seq_length)
  #   all_results = list(tqdm(p.imap(func_, )))
