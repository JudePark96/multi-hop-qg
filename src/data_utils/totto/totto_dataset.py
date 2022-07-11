import json
import logging
import os
import sys
from functools import partial
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, T5Tokenizer

sys.path.append(os.getcwd() + "/../")  # noqa: E402

from src.data_utils.totto.preprocess_utils import get_highlighted_subtable, linearize_subtable

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class TottoDataset(Dataset):

  def __init__(self, data_path: str) -> None:
    super().__init__()
    self.examples = self.read_jsonl(data_path)

  def __getitem__(self, idx: int) -> Dict[str, str]:
    example = self.examples[idx]
    return example

  def __len__(self) -> int:
    return len(self.examples)

  def read_jsonl(self, path: str) -> List[Dict[str, Any]]:
    with open(path, 'r') as f:
      examples = []
      for idx, i in tqdm(enumerate(f), desc='reading the jsonl ...'):
        try:
          example = json.loads(i)
          subtable = get_highlighted_subtable(table=example['table'], cell_indices=example['highlighted_cells'],
                                              with_heuristic_headers=True)
          cells_linearized = linearize_subtable(
            subtable=subtable,
            table_page_title=example['table_page_title'],
            table_section_title=example['table_section_title']
          )

          gold_sentence = example['sentence_annotations'][0]['final_sentence']

          examples.append({
            'input_sentence': f'generate table description: {cells_linearized}',
            'gold_sentence': gold_sentence
          })
        except Exception as e:
          logger.info(e)

    return examples

  @staticmethod
  def collate_fn(batch: List[Dict[str, str]], tokenizer: T5Tokenizer, max_len: int = 512):
    input_sentences = [b['input_sentence'] for b in batch]
    gold_sentences = [b['gold_sentence'] for b in batch]

    input_sentences = tokenizer(input_sentences, max_length=max_len, padding='max_length', truncation=True)
    input_ids = torch.tensor(input_sentences['input_ids'])
    attention_mask = torch.tensor(input_sentences['attention_mask'])

    labels = []

    for sentence in gold_sentences:
      label = tokenizer.encode(sentence)[:max_len]
      while len(label) < max_len:
        label.append(-100)

      labels.append(label)

    labels = torch.tensor(labels)

    return {
      'input_ids': input_ids,
      'attention_mask': attention_mask,
      'labels': labels
    }


if __name__ == '__main__':
  tokenizer = T5Tokenizer.from_pretrained('t5-base')
  max_len = 1024
  dataset = TottoDataset('../resources/totto/totto_train_data.jsonl')

  loader = DataLoader(dataset, batch_size=4, collate_fn=partial(TottoDataset.collate_fn, tokenizer=tokenizer))

  for batch in loader:
    print(batch)
    exit()
