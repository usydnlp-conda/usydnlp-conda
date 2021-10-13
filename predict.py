import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from utils import init_logger, load_tokenizer, get_intent_labels, get_slot_labels, MODEL_CLASSES
logger = logging.getLogger(__name__)


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, 'training_args.bin'))


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = MODEL_CLASSES[args.model_type][1].from_pretrained(args.model_dir,
                                                                  args=args,
                                                                  intent_label_lst=get_intent_labels(args),
                                                                  slot_label_lst=get_slot_labels(args))
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def read_input_file(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    return lines


def convert_input_file_to_tensor_dataset(lines,
                                         pred_config,
                                         args,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []
    all_ner_embeds = []
    
    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)
        
        # additional embeddings
        import pickle
        
        # ner embedding
        if 'd' in args.add_embed:
            if args.task=='conda' and args.add_embed=='dg':
                try:
                    type(all_vocab)
                except:
                    if args.task == "conda":
                        ner_file = open("data/conda/embed_size_50/conda_ner_embed.pkl", "rb")
                    else:
                        ner_file = open("data/low/embed_size_50/lol_ner_embed.pkl", "rb")
                    all_vocab = pickle.load(ner_file)
                    ner_file.close()
                if len(words)<args.max_seq_len:
                  ner_embed = []
                  ner_embed.append(all_vocab['start'])
                  for word in words:
                    word_tokens = tokenizer.tokenize(word)
                    for i in range(len(word_tokens)):
                        try:
                          ner_embed.append(all_vocab[word.lower()]*(i+1)) # to cater for bert tokenization
                        except:
                          ner_embed.append(np.zeros(9))
                  ner_embed.append(all_vocab['end'])
                  while len(ner_embed)<args.max_seq_len:
                    ner_embed.append(np.zeros(9))
                else:
                  ner_embed = []
                  ner_embed.append(all_vocab['start'])
                  for word in words[:args.max_seq_len]:
                    word_tokens = tokenizer.tokenize(word)
                    for i in range(len(word_tokens)):
                        try:
                          ner_embed.append(all_vocab[word.lower()]*(i+1)) # to cater for bert tokenization
                        except:
                          ner_embed.append(np.zeros(9))
                  ner_embed.append(all_vocab['end'])
                ner_embed = np.array(ner_embed)
                if ner_embed.shape[0]>args.max_seq_len:
                    ner_embed = ner_embed[:args.max_seq_len,:]
            else:
                try:
                    type(all_vocab)
                except:
                    if args.task == "conda":
                        ner_file = open("data/conda/conda_ner_embed_jbert.pkl", "rb")
                    else:
                        ner_file = open("data/low/lol_ner_embed_jbert.pkl", "rb")
                    all_vocab = pickle.load(ner_file)
                    ner_file.close()
                if len(tokens)<args.max_seq_len:
                  ner_embed = []
                  for word in tokens:
                    try:
                      ner_embed.append(all_vocab[word])
                    except:
                      ner_embed.append(np.zeros(9))
                  while len(ner_embed)<args.max_seq_len:
                    ner_embed.append(np.zeros(9))
                else:
                  ner_embed = []
                  for word in tokens[:args.max_seq_len]:
                    try:
                      ner_embed.append(all_vocab[word])
                    except:
                      ner_embed.append(np.zeros(9))
                ner_embed = np.array(ner_embed)
                if ner_embed.shape[0]>args.max_seq_len:
                    ner_embed = ner_embed[:args.max_seq_len,:]
        
        # glove embedding
        if 'g' in args.add_embed:
            try:
                type(glove_embed)
            except:
                if args.task == "conda":
                    glove_file = open("data/conda/conda_glove_embed.pkl", "rb")
                else:
                    glove_file = open("data/low/lol_glove_embed.pkl", "rb")
                glove_embed = pickle.load(glove_file)
                glove_file.close()
            
            if len(words)<args.max_seq_len:
              unbiglv_embed = []
              unbiglv_embed.append(glove_embed['start'])
              for word in words:
                  word_tokens = tokenizer.tokenize(word)
                  for i in range(len(word_tokens)):
                      try:
                          unbiglv_embed.append(glove_embed[word.lower()]*(i+1)) # to cater for bert tokenization
                      except:
                          unbiglv_embed.append(np.zeros(300))
              unbiglv_embed.append(glove_embed['end'])
              while len(unbiglv_embed)<args.max_seq_len:
                unbiglv_embed.append(np.zeros(300))
            else:
              unbiglv_embed = []
              unbiglv_embed.append(glove_embed['start'])
              for word in words[:args.max_seq_len]:
                  word_tokens = tokenizer.tokenize(word)
                  for i in range(len(word_tokens)):
                      try:
                          unbiglv_embed.append(glove_embed[word.lower()]*(i+1)) # to cater for bert tokenization
                      except:
                          unbiglv_embed.append(np.zeros(300))
              unbiglv_embed.append(glove_embed['end'])
            unbiglv_embed = np.array(unbiglv_embed)
            if unbiglv_embed.shape[0]>args.max_seq_len:
                unbiglv_embed = unbiglv_embed[:args.max_seq_len,:]
            
        # dict2vec embedding
        if 'e' in args.add_embed:
            try:
                type(dict2vec_embed)
            except:
                if args.task == "conda":
                    dict2vec_file = open("data/conda/conda_dict2vec_embed.pkl", "rb")
                else:
                    dict2vec_file = open("data/low/lol_dict2vec_embed.pkl", "rb")
                d2v_embed = pickle.load(dict2vec_file)
                dict2vec_file.close()
            
            if len(words)<args.max_seq_len:
              dict2vec_embed = []
              dict2vec_embed.append(d2v_embed['start'])
              for word in words:
                  word_tokens = tokenizer.tokenize(word)
                  for i in range(len(word_tokens)):
                      try:
                          dict2vec_embed.append(d2v_embed[word.lower()]*(i+1)) # to cater for bert tokenization
                      except:
                          dict2vec_embed.append(np.zeros(300))
              dict2vec_embed.append(d2v_embed['end'])
              while len(dict2vec_embed)<args.max_seq_len:
                dict2vec_embed.append(np.zeros(300))
            else:
              dict2vec_embed = []
              dict2vec_embed.append(d2v_embed['start'])
              for word in words[:args.max_seq_len]:
                  word_tokens = tokenizer.tokenize(word)
                  for i in range(len(word_tokens)):
                      try:
                          dict2vec_embed.append(d2v_embed[word.lower()]*(i+1)) # to cater for bert tokenization
                      except:
                          dict2vec_embed.append(np.zeros(300))
              dict2vec_embed.append(d2v_embed['end'])
            dict2vec_embed = np.array(dict2vec_embed)
            if dict2vec_embed.shape[0]>args.max_seq_len:
                dict2vec_embed = dict2vec_embed[:args.max_seq_len,:]
        
        if args.add_embed=='None':
            ner_embed= np.zeros((args.max_seq_len,1))
        elif args.add_embed=='dg':
            ner_embed = np.concatenate((ner_embed,unbiglv_embed),axis=1)
        elif args.add_embed=='de':
            ner_embed = np.concatenate((ner_embed,dict2vec_embed),axis=1)
        elif args.add_embed=='g':
            ner_embed = unbiglv_embed
        elif args.add_embed=='e':
            ner_embed = dict2vec_embed
        elif args.add_embed!='d':
            print('Only the following combinations can be chosen for --add_embed input: d,g,e,dg,de.')
        
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)
        all_ner_embeds.append(ner_embed)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)
    all_ner_embeds = torch.tensor(all_ner_embeds, dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask,all_ner_embeds)

    return dataset


def predict(pred_config):
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    logger.info(args)

    intent_label_lst = get_intent_labels(args)
    slot_label_lst = get_slot_labels(args)

    # Convert input file to TensorDataset
    pad_token_label_id = args.ignore_index
    tokenizer = load_tokenizer(args)
    lines = read_input_file(pred_config)
    dataset = convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    all_slot_label_mask = None
    intent_preds = None
    slot_preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad(): # disable gradient calculation 
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "intent_label_ids": None,
                      "slot_labels_ids": None,
                      'ner_embed': batch[4]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]

            # Intent Prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                if args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    intent_preds = np.argmax(intent_preds, axis=1)

    if not args.use_crf:
        slot_preds = np.argmax(slot_preds, axis=2)

    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    slot_preds_list = [[] for _ in range(slot_preds.shape[0])]

    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    # Write to output file
    with open(pred_config.output_file, "w", encoding="utf-8") as f:
        for words, slot_preds, intent_pred in zip(lines, slot_preds_list, intent_preds):
            line = ""
            for word, pred in zip(words, slot_preds):
                if pred == 'O':
                    line = line + word + " "
                else:
                    line = line + "[{}:{}] ".format(word, pred)
            f.write("<{}> -> {}\n".format(intent_label_lst[intent_pred], line.strip()))

    logger.info("Prediction Done!")


if __name__ == "__main__":
    init_logger() # configures the root logger - see utils.py
    parser = argparse.ArgumentParser() # create an ArgumentParser object

    # add arguments to the ArgumentParser object
    parser.add_argument("--input_file", default="sample_pred_in.txt", type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="sample_pred_out.txt", type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", default="./atis_model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    # parse the arguments, convert each argument to the appropriate type and then invoke the appropriate action
    pred_config = parser.parse_args()
    
    # call the predict function above on line 129
    predict(pred_config)
