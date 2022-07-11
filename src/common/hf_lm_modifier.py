import logging

from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
  # model_name_or_config_path = 't5-base'
  model_name_or_config_path = 'facebook/bart-large'

  if 't5' in model_name_or_config_path:
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_config_path)
    tokenizer = T5Tokenizer.from_pretrained(model_name_or_config_path)

    print(len(tokenizer))
    tokenizer.add_special_tokens({'additional_special_tokens': ['<ANS>', '</ANS>', '<p>', '</p>']})
    print(len(tokenizer))

    model.resize_token_embeddings(len(tokenizer))

    tokenizer.save_pretrained('../../resources/t5-base-add-tokens-{ans,p}')
    model.save_pretrained('../../resources/t5-base-add-tokens-{ans,p}')
  elif 'bart' in model_name_or_config_path:
    model = BartForConditionalGeneration.from_pretrained(model_name_or_config_path)
    tokenizer = BartTokenizer.from_pretrained(model_name_or_config_path)

    print(len(tokenizer))
    tokenizer.add_special_tokens({'additional_special_tokens': ['<ANS>', '</ANS>', '<p>', '</p>']})
    print(len(tokenizer))

    model.resize_token_embeddings(len(tokenizer))

    tokenizer.save_pretrained(f'../../resources/{model_name_or_config_path.split("/")[-1]}-add-tokens')
    model.save_pretrained(f'../../resources/{model_name_or_config_path.split("/")[-1]}-add-tokens')