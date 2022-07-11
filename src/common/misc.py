import gzip
import logging
import pickle
from typing import Any

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def save_pkl_binary(path: str, obj: Any) -> None:
  with gzip.open(path, 'wb') as f:
    pickle.dump(obj, f)


def read_pkl_binary(path: str) -> Any:
  with gzip.open(path, 'rb') as f:
    data = pickle.load(f)

  return data


def intersection(a: str, b: str) -> bool:
  a, b = a.strip(), b.strip()

  flag = False

  if len(a) > len(b):
    if b in a:
      flag = True
  elif len(b) > len(a):
    if a in b:
      flag = True
  elif len(a) == len(b):
    if a in b and b in a:
      flag = True

  return flag


def print_args(params):
  logger.info(" **************** CONFIGURATION **************** ")
  for key, val in vars(params).items():
    key_str = "{}".format(key) + (" " * (30 - len(key)))
    logger.info("%s =   %s", key_str, val)
