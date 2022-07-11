import logging
from dataclasses import dataclass
from typing import List, Tuple

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample(object):
  qid: str
  qas_type: str
  question_tokens: List[str]
  answer_tokens: List[str]
  paragraph_tokens: List[List[str]]
  coref_positions: List[Tuple[int, int]]
  entity_start_end_positions: List[List[Tuple[int, int]]]


@dataclass(frozen=True)
class InputFeature(object):
  qid: str
  input_ids: List[int]  # <ANS> Answer </ANS> <SEP> Sequence <SEP>
  decoder_input_ids: List[int]
  attention_mask: List[int]
  labels: List[int]
  entity_start_end_position: List[Tuple[int, int]]
  coref_position: List[Tuple[int, int]]

