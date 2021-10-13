import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel, BertConfig
from torchcrf import CRF # https://pytorch-crf.readthedocs.io/en/stable/
from .module import IntentClassifier, SlotClassifier


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert
        
        # initiate IntentClassifier and SlotClassifier from module.py
        add_embed_options = {'None':0,'d':9,'g':300,'e':300,'dg':309,'de':309} # additional input_dim for additional embeddings
        input_dim = config.hidden_size+add_embed_options[args.add_embed]
        self.intent_classifier = IntentClassifier(input_dim, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(input_dim, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            # add crf on top of bert
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids, ner_embed):
    
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, 
                            output_hidden_states=False, 
                            output_attentions=False)  
        
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]
        
        # adding additional embeddings
        if self.args.add_embed!='None':
            ner_embed_squeezed = torch.mean(ner_embed,dim=1).to(self.device)
            pooled_output = torch.cat((pooled_output,ner_embed_squeezed), dim=1).to(self.device)
            sequence_output = torch.cat((sequence_output,ner_embed), dim=2)
        
        intent_logits = self.intent_classifier(pooled_output) # see module.py
        slot_logits = self.slot_classifier(sequence_output) # see module.py
        
        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None: # for training and evaluation
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else: 
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss
        
        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here
        
        outputs = (total_loss,) + outputs
        
        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits`1
