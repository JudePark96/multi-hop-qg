import argparse
import json
import logging
import os

import nltk

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_path', type=str)
  parser.add_argument('--input_file', type=str)
  parser.add_argument('--output_file', type=str)

  return parser.parse_args()


def word_tokenize(tokens):
  return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


if __name__ == '__main__':
  args = get_arguments()

  input_path = os.path.join(args.input_path, args.input_file)

  with open(input_path, 'rb') as f:
    generated_output = json.load(f)

  ref = [i['label'] for i in generated_output]
  hyp = [i['generated_question'] for i in generated_output]

  output_ref_path = os.path.join(args.input_path, f'ref_{args.output_file}')
  output_hyp_path = os.path.join(args.input_path, f'hyp_{args.output_file}')

  with open(output_ref_path, 'a+', encoding='utf-8') as f:
    for i in ref:
      f.write(' '.join(word_tokenize(i)).strip() + '\n')

  with open(output_hyp_path, 'a+', encoding='utf-8') as f:
    for i in hyp:
      f.write(' '.join(word_tokenize(i)).strip() + '\n')
