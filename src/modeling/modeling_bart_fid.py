"""
We have forked some part of codes from << https://github.com/facebookresearch/FiD/blob/main/src/model.py >>
"""

import logging
import os
import sys
from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BartTokenizer
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutput

sys.path.append(os.getcwd() + "/../")  # noqa: E402

from src.common.misc import read_pkl_binary
from src.data_utils.hotpotqa.dto_wrapper import InputFeature
from src.data_utils.hotpotqa.feature_util import read_hotpotqa_fid_examples, convert_features_to_dataset
from src.modeling.config.configuration_bart import BartConfig
from src.modeling.modeling_bart import BartModel, shift_tokens_right, BartForConditionalGeneration

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt="%m/%d/%Y %H:%M:%S",
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def reformat_encoder_outputs_to_fid_format(
  encoder_outputs: BaseModelOutput,
  bsz: int,
  n_passages: int,
  seq_length: int,
) -> BaseModelOutput:
  # [bsz * n_passages x seq_length x dim]
  last_hidden_state = encoder_outputs.last_hidden_state
  last_hidden_state = last_hidden_state.view(bsz, n_passages * seq_length, -1)

  hidden_states = None if encoder_outputs.hidden_states is None else encoder_outputs.hidden_states
  attentions = None if encoder_outputs.attentions is None else encoder_outputs.attentions

  if hidden_states is not None:
    hidden_states = *(h.view(bsz, n_passages * seq_length, -1) for h in hidden_states),

  if attentions is not None:
    attentions = *(a.view(bsz, a.size(1), n_passages * seq_length, -1) for a in attentions),

  return BaseModelOutput(
    last_hidden_state=last_hidden_state,
    hidden_states=hidden_states,
    attentions=attentions
  )


class FidBartModel(BartModel):
  def __init__(self, config: BartConfig):
    super().__init__(config)
    assert self.config.strategy in ['fid', 'fid_graph', 'normal']

  def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    decoder_attention_mask: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    decoder_head_mask: Optional[torch.Tensor] = None,
    cross_attn_head_mask: Optional[torch.Tensor] = None,
    encoder_outputs: Optional[List[torch.FloatTensor]] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
  ) -> Union[Tuple, Seq2SeqModelOutput]:
    # different to other models, Bart automatically creates decoder_input_ids from
    # input_ids if no decoder_input_ids are provided
    if decoder_input_ids is None and decoder_inputs_embeds is None:
      if input_ids is None:
        raise ValueError(
          "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
          "passed, `input_ids` cannot be `None`. Please pass either "
          "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
        )

      decoder_input_ids = shift_tokens_right(
        input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
      )

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
      output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    bsz, n_passages, seq_length = input_ids.shape
    if encoder_outputs is None:
      if not self.config.strategy == 'normal':
        input_ids = input_ids.view(bsz * n_passages, -1)
        if attention_mask is not None:
          attention_mask = attention_mask.view(bsz * n_passages, -1)

        if self.config.strategy == 'fid_graph':
          # TODO: FID + Graph 이면 Graph 관련 연산을 추가해주어야함.
          pass

      encoder_outputs = self.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
      )
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
      encoder_outputs = BaseModelOutput(
        last_hidden_state=encoder_outputs[0],
        hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
      )

    encoder_outputs = reformat_encoder_outputs_to_fid_format(encoder_outputs, bsz, n_passages, seq_length)
    print('encoder_outputs', encoder_outputs.last_hidden_state.shape)

    decoder_outputs = self.decoder(
      input_ids=decoder_input_ids,
      attention_mask=decoder_attention_mask,
      encoder_hidden_states=encoder_outputs[0],
      encoder_attention_mask=attention_mask,
      head_mask=decoder_head_mask,
      cross_attn_head_mask=cross_attn_head_mask,
      past_key_values=past_key_values,
      inputs_embeds=decoder_inputs_embeds,
      use_cache=use_cache,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    if not return_dict:
      return decoder_outputs + encoder_outputs

    return Seq2SeqModelOutput(
      last_hidden_state=decoder_outputs.last_hidden_state,
      past_key_values=decoder_outputs.past_key_values,
      decoder_hidden_states=decoder_outputs.hidden_states,
      decoder_attentions=decoder_outputs.attentions,
      cross_attentions=decoder_outputs.cross_attentions,
      encoder_last_hidden_state=encoder_outputs.last_hidden_state,
      encoder_hidden_states=encoder_outputs.hidden_states,
      encoder_attentions=encoder_outputs.attentions,
    )


class FidBartForConditionalGeneration(BartForConditionalGeneration):
  def __init__(self, config: BartConfig):
    super().__init__(config)
    self.model = FidBartModel(config)


if __name__ == '__main__':
  model = FidBartForConditionalGeneration.from_pretrained('../../resources/fid-bart-base-add-tokens')
  tokenizer = BartTokenizer.from_pretrained('../../resources/fid-bart-base-add-tokens')

  input_features = read_pkl_binary('../../resources/hotpotqa/preprocessed/bart_train_features.pkl.gz')
  dataset = convert_features_to_dataset(input_features)
  loader = DataLoader(dataset, batch_size=4, shuffle=False)

  for batch in loader:
    input_ids, attention_mask, decoder_input_ids, labels = batch[0], batch[1], batch[2], batch[3]
    print(input_ids.shape, attention_mask.shape, decoder_input_ids.shape)
    # exit()
    output = model(input_ids=input_ids,
                   attention_mask=attention_mask,
                   decoder_input_ids=decoder_input_ids)
    print(output)
    exit()

  # input = torch.randint(100, (32, 2, 128))
  # input_ = torch.randint(100, (32, 128))
  #
  # print(input.shape)
  # output = model(input_ids=input, decoder_input_ids=input_)
  # print(output)
