import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from utils import MODEL_CLASSES, compute_metrics, get_intent_labels, get_slot_labels

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.intent_label_lst = get_intent_labels(args) # get the intent labels - see utils.py
        self.slot_label_lst = get_slot_labels(args) # get the slot labels - see utils.py
        
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index # set pad_token_label_id to 0

        # the below sets config and model classes - see utils.py
        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args,
                                                      intent_label_lst=self.intent_label_lst,
                                                      slot_label_lst=self.slot_label_lst)
        
        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(torch.cuda.current_device()))
        print('Using device:', self.device)
        self.model.to(self.device)
        
        # set the initial parameters for best model
        self.best_metric = -0.1
        self.best_result = {}
        self.best_model = False

    def train(self):
        # sample data for training 
        train_sampler = RandomSampler(self.train_dataset)
        
        # load the data
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs # // is floor division

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], # get all parameters excluding ['bias', 'LayerNorm.weight']
             'weight_decay': self.args.weight_decay}, # default args.weight_decay is 0.0
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} # get parameters for ['bias', 'LayerNorm.weight']
        ]
              
        # implements Adam algorithm with weight decay fix as introduced in Decoupled Weight Decay Regularization
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon) 
        
        # get_linear_schedule_with_warmup creates a schedule with a learning rate that decreases linearly 
        # from the initial lr set in the optimizer to 0, after a warmup period during 
        # which it increases linearly from 0 to the initial lr set in the optimizer
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)
        
        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad() # set the gradients to zero
        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            
            for step, batch in enumerate(epoch_iterator): # iterate over the train_dataloader
                
                self.model.train() # set to train mode
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                
                # set the inputs
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4],
                          'ner_embed': batch[5]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                
                outputs = self.model(**inputs) # compute the outputs
                loss = outputs[0] 

                if self.args.gradient_accumulation_steps > 1: # default is 1 - see main.py
                    loss = loss / self.args.gradient_accumulation_steps
                
                # loss.backward() computes dloss/dx for every parameter x which has requires_grad=True
                # these are accumulated into x.grad for every parameter x
                loss.backward() # compute the gradients
                
                # the item() method extracts the loss’s value as a Python float
                # the below calculates the running loss so that we can calculate the mean loss of that epoch later
                tr_loss += loss.item()
                
                # % is Modulus operator, it gives the remainder of the division of left operand by the right
                if (step + 1) % self.args.gradient_accumulation_steps == 0: 
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    
                    self.model.zero_grad()
                    global_step += 1
                    
                    # default args.logging_steps is 200 - evaulate on dev set every 200 steps
                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate("dev")
                    #self.evaluate("dev") ### FOR TESTING ONLY, TO DELETE FOR NORMAL DATA ###
                    
                    # default args.save_steps is 200 - save every 200 steps
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        # save the model if the current optimised metric is higher (if not set, it will save at every 'dev' evaluation)
                        if self.best_model==True:
                            self.save_model()
                            print("Best model saved")
                            self.best_model = False
                    #self.save_model() ### FOR TESTING ONLY, TO DELETE FOR NORMAL DATA ###
                    
                if 0 < self.args.max_steps < global_step: # default args.max_steps is -1
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")
        
        eval_sampler = SequentialSampler(dataset)
        
        # load the data using pytorch DataLoader
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        self.model.eval() # set to eval mode

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad(): # disable gradient calculation 
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4],
                          'ner_embed': batch[5]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                # outputs include (loss), logits, (hidden_states), (attentions)
                # logits is a tuple of intent and slot logits`1
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2] 
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0)
        
        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)
        
        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        # calculate result metrics - see utils.py line 64 for compute_metrics function
        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list, self.args)
        
        # code to optimise based on optim_metric parameter
        if self.args.optim_metric!='None':
            if self.args.optim_metric=='JSA':
                current_metric = total_result['semantic_frame_acc']
            elif self.args.optim_metric=='UCA':
                current_metric = total_result['intent_acc']
            else:
                current_metric = total_result[self.args.optim_metric]
                
            if current_metric>self.best_metric:
                self.best_metric = current_metric
                self.best_result = total_result
                self.best_model = True
        else:
            self.best_model = True
        
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        
        # show info for current best evaluation results
        if mode == 'dev':
            print()
            logger.info("***** Current best eval results based on " + self.args.optim_metric +" *****")
            for key in sorted(self.best_result.keys()):
                logger.info("  %s = %s", key, str(self.best_result[key]))
        
        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args,
                                                          intent_label_lst=self.intent_label_lst,
                                                          slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")