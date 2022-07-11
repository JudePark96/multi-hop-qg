# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
We have forked some part of codes from << https://github.com/facebookresearch/FiD/blob/main/src/model.py >>
"""
import os
import sys
import types
from typing import Union, Tuple, Optional

import torch
import transformers
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import numpy as np
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

sys.path.append(os.getcwd() + "/../")  # noqa: E402

from src.common.misc import read_pkl_binary
from src.data_utils.hotpotqa.feature_util import convert_features_to_dataset
from src.modeling.config.configuration_t5 import T5Config
from src.modeling.modeling_t5 import T5ForConditionalGeneration


class FiDT5(T5ForConditionalGeneration):
  def __init__(self, config: T5Config):
    super().__init__(config)

  def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
    if head_mask is not None and decoder_head_mask is None:
      if self.config.num_layers == self.config.num_decoder_layers:
        # warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
        decoder_head_mask = head_mask

    if input_ids != None:
      if input_ids.dim() == 3:
        bsz, n_passages, seq_length = input_ids.shape
        input_ids = input_ids.view(bsz * n_passages, seq_length)
        if attention_mask is not None:
          attention_mask = attention_mask.view(bsz * n_passages, seq_length)

    # Encode if needed (training, first prediction pass)
    if encoder_outputs is None:
      # Convert encoder inputs in embeddings if needed
      encoder_outputs = self.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        head_mask=head_mask,
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

    hidden_states = encoder_outputs[0]

    if self.config.strategy in ['fid', 'fid_graph']:
      if self.training:
        n_passages = self.config.n_passages
        bsz = hidden_states.size(0) // n_passages
        seq_length = hidden_states.size(1)
        hidden_states = hidden_states.view(bsz, n_passages, seq_length, -1)
        hidden_states = hidden_states.view(bsz, n_passages * seq_length, -1)

        if attention_mask is not None:
          attention_mask = attention_mask.view(bsz, n_passages, seq_length)
          attention_mask = attention_mask.view(bsz, n_passages * seq_length)
      pass

    if self.model_parallel:
      torch.cuda.set_device(self.decoder.first_device)

    if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
      # get decoder inputs from shifting lm labels to the right
      decoder_input_ids = self._shift_right(labels)

    # Set device for model parallelism
    if self.model_parallel:
      torch.cuda.set_device(self.decoder.first_device)
      hidden_states = hidden_states.to(self.decoder.first_device)
      if decoder_input_ids is not None:
        decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
      if attention_mask is not None:
        attention_mask = attention_mask.to(self.decoder.first_device)
      if decoder_attention_mask is not None:
        decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

    # Decode
    decoder_outputs = self.decoder(
      input_ids=decoder_input_ids,
      attention_mask=decoder_attention_mask,
      inputs_embeds=decoder_inputs_embeds,
      past_key_values=past_key_values,
      encoder_hidden_states=hidden_states,
      encoder_attention_mask=attention_mask,
      head_mask=decoder_head_mask,
      cross_attn_head_mask=cross_attn_head_mask,
      use_cache=use_cache,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    sequence_output = decoder_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
      torch.cuda.set_device(self.encoder.first_device)
      self.lm_head = self.lm_head.to(self.encoder.first_device)
      sequence_output = sequence_output.to(self.lm_head.weight.device)

    if self.config.tie_word_embeddings:
      # Rescale output before projecting on vocab
      # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
      sequence_output = sequence_output * (self.model_dim ** -0.5)

    lm_logits = self.lm_head(sequence_output)

    loss = None
    if labels is not None:
      loss_fct = CrossEntropyLoss(ignore_index=-100)
      loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
      # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

    if not return_dict:
      output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
      return ((loss,) + output) if loss is not None else output

    return Seq2SeqLMOutput(
      loss=loss,
      logits=lm_logits,
      past_key_values=decoder_outputs.past_key_values,
      decoder_hidden_states=decoder_outputs.hidden_states,
      decoder_attentions=decoder_outputs.attentions,
      cross_attentions=decoder_outputs.cross_attentions,
      encoder_last_hidden_state=encoder_outputs.last_hidden_state,
      encoder_hidden_states=encoder_outputs.hidden_states,
      encoder_attentions=encoder_outputs.attentions,
    )

  # def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
  #   if input_ids.dim() == 3:
  #     bsz, n_passages, seq_length = input_ids.shape
  #     input_ids = input_ids.view(bsz * n_passages, seq_length)
  #     if attention_mask is not None:
  #       attention_mask = attention_mask.view(bsz * n_passages, seq_length)
  #
  #     return super().generate(
  #       input_ids=input_ids,
  #       attention_mask=attention_mask,
  #       **kwargs
  #     )


if __name__ == '__main__':
  model = FiDT5.from_pretrained('../../resources/fid-t5-base-add-tokens')
  tokenizer = T5Tokenizer.from_pretrained('../../resources/fid-t5-base-add-tokens')

  input_features = read_pkl_binary('../../resources/hotpotqa/preprocessed/t5_train_features.pkl.gz')
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
