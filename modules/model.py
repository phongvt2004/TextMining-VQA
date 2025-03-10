import torch
import torch.nn as nn
from transformers import ViltForQuestionAnswering
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import List, Optional, Tuple, Union

class CustomViltForVQA(ViltForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)
        self.loss_fn = nn.CrossEntropyLoss()  # Use CrossEntropyLoss
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooler_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooler_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable PP
            labels = labels.to(logits.device)
            loss = self.loss_fn(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )