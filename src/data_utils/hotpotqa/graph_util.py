import logging
import os
import sys

import numpy as np
import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.getcwd() + "/../")  # noqa: E402

from src.common.misc import read_pkl_binary
from src.data_utils.hotpotqa.dto_wrapper import InputExample


def create_entity_graph(
  last_hidden_states: torch.Tensor
) -> None:
  pass


if __name__ == '__main__':
  path = '../../../resources/hotpotqa/preprocessed/bart_dev_examples_maxlen384.pkl.gz'
  data = read_pkl_binary(path)

  for idx, case in enumerate(data):
    create_entity_graph(case, 40)
    if idx == 4:
      exit()
