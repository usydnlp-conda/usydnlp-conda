import os
import copy
import json
import logging
import numpy as np

import torch
from torch.utils.data import TensorDataset

from utils import get_intent_labels, get_slot_labels

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, slot_labels_ids, ner_embed):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids
        self.ner_embed = ner_embed

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args) # get the intent labels - see utils.py
        self.slot_labels = get_slot_labels(args) # get the slot labels - see utils.py

        self.input_text_file = 'seq.in'
        self.intent_label_file = 'label'
        self.slot_labels_file = 'seq.out'

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, intents, slots, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (text, intent, slot) in enumerate(zip(texts, intents, slots)):
            guid = "%s-%s" % (set_type, i)
            # 1. input_text
            words = text.split()  # Some are spaced twice
            # 2. intent
            intent_label = self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            # 3. slot
            slot_labels = []
            for s in slot.split():
                slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))
            
            # identify if lengths are not the same
            if len(words) != len(slot_labels):              
                print("index:",i)                           
                print("len(words):",len(words))             
                print("len(slot_labels):",len(slot_labels)) 
                print("words:",words)                       
            assert len(words) == len(slot_labels)
            examples.append(InputExample(guid=guid, words=words, intent_label=intent_label, slot_labels=slot_labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)), # read input_text_file ('seq.in'))
                                     intents=self._read_file(os.path.join(data_path, self.intent_label_file)), # read intent_label_file ('label')
                                     slots=self._read_file(os.path.join(data_path, self.slot_labels_file)), # read slot_labels_file ('seq.out')
                                     set_type=mode)

processors = {
    "atis": JointProcessor,
    "snips": JointProcessor,
    "conda": JointProcessor, 
    "low": JointProcessor
}


def convert_examples_to_features(mode, args, examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), max_seq_len)

        intent_label_id = int(example.intent_label)
        
        # additional embeddings
        import pickle
        
        # ner embedding
        if 'd' in args.add_embed:
            if args.task=='conda' and args.add_embed=='dg':
                try:
                    type(all_vocab) # if already exist, no need to reload
                except:
                    if args.task == "conda":
                        ner_file = open("data/conda/embed_size_50/conda_ner_embed.pkl", "rb")
                    else:
                        ner_file = open("data/low/embed_size_50/lol_ner_embed.pkl", "rb")
                    all_vocab = pickle.load(ner_file)
                    ner_file.close()
                if len(example.words)<max_seq_len:
                  ner_embed = []
                  ner_embed.append(all_vocab['start']) # to account for the start token
                  for word in example.words:
                    word_tokens = tokenizer.tokenize(word)
                    # note: bert tokenizer can break down words into subwords, the code below is to cater for that
                    # if a word is broken down into subwords, the embedding for that word will be repeated that many times 
                    # with a slight difference for each repeat, using the following formula: *(i+1)
                    for i in range(len(word_tokens)):
                        try:
                          ner_embed.append(all_vocab[word.lower()]*(i+1)) # to cater for bert tokenization
                        except:
                          ner_embed.append(np.zeros(9))
                  ner_embed.append(all_vocab['end']) # to account for the end token
                  while len(ner_embed)<max_seq_len:
                    ner_embed.append(np.zeros(9))
                else:
                  ner_embed = []
                  ner_embed.append(all_vocab['start'])
                  for word in example.words[:max_seq_len]:
                    word_tokens = tokenizer.tokenize(word)
                    for i in range(len(word_tokens)):
                        try:
                          ner_embed.append(all_vocab[word.lower()]*(i+1)) # to cater for bert tokenization
                        except:
                          ner_embed.append(np.zeros(9))
                  ner_embed.append(all_vocab['end'])
                ner_embed = np.array(ner_embed)
                if ner_embed.shape[0]>max_seq_len:
                    ner_embed = ner_embed[:max_seq_len,:]
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
                if len(tokens)<max_seq_len:
                  ner_embed = []
                  for word in tokens:
                    try:
                      ner_embed.append(all_vocab[word])
                    except:
                      ner_embed.append(np.zeros(9))
                  while len(ner_embed)<max_seq_len:
                    ner_embed.append(np.zeros(9))
                else:
                  ner_embed = []
                  for word in tokens[:max_seq_len]:
                    try:
                      ner_embed.append(all_vocab[word])
                    except:
                      ner_embed.append(np.zeros(9))
                ner_embed = np.array(ner_embed)
                if ner_embed.shape[0]>max_seq_len:
                    ner_embed = ner_embed[:max_seq_len,:]
        
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
            
            if len(example.words)<max_seq_len:
              unbiglv_embed = []
              unbiglv_embed.append(glove_embed['start'])
              for word in example.words:
                  word_tokens = tokenizer.tokenize(word)
                  for i in range(len(word_tokens)):
                      try:
                          unbiglv_embed.append(glove_embed[word.lower()]*(i+1)) # to cater for bert tokenization
                      except:
                          unbiglv_embed.append(np.zeros(300))
              unbiglv_embed.append(glove_embed['end'])
              while len(unbiglv_embed)<max_seq_len:
                unbiglv_embed.append(np.zeros(300))
            else:
              unbiglv_embed = []
              unbiglv_embed.append(glove_embed['start'])
              for word in example.words[:max_seq_len]:
                  word_tokens = tokenizer.tokenize(word)
                  for i in range(len(word_tokens)):
                      try:
                          unbiglv_embed.append(glove_embed[word.lower()]*(i+1)) # to cater for bert tokenization
                      except:
                          unbiglv_embed.append(np.zeros(300))
              unbiglv_embed.append(glove_embed['end'])
            unbiglv_embed = np.array(unbiglv_embed)
            if unbiglv_embed.shape[0]>max_seq_len:
                unbiglv_embed = unbiglv_embed[:max_seq_len,:]
            
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
            
            if len(example.words)<max_seq_len:
              dict2vec_embed = []
              dict2vec_embed.append(d2v_embed['start'])
              for word in example.words:
                  word_tokens = tokenizer.tokenize(word)
                  for i in range(len(word_tokens)):
                      try:
                          dict2vec_embed.append(d2v_embed[word.lower()]*(i+1))
                      except:
                          dict2vec_embed.append(np.zeros(300))
              dict2vec_embed.append(d2v_embed['end'])
              while len(dict2vec_embed)<max_seq_len:
                dict2vec_embed.append(np.zeros(300))
            else:
              dict2vec_embed = []
              dict2vec_embed.append(d2v_embed['start'])
              for word in example.words[:max_seq_len]:
                  word_tokens = tokenizer.tokenize(word)
                  for i in range(len(word_tokens)):
                      try:
                          dict2vec_embed.append(d2v_embed[word.lower()]*(i+1))
                      except:
                          dict2vec_embed.append(np.zeros(300))
              dict2vec_embed.append(d2v_embed['end'])
            dict2vec_embed = np.array(dict2vec_embed)
            if dict2vec_embed.shape[0]>max_seq_len:
                dict2vec_embed = dict2vec_embed[:max_seq_len,:]
        
        # concat the final ner_embed depending on which ones are chosen
        if args.add_embed=='None':
            ner_embed= np.zeros((max_seq_len,1))
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
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))
            logger.info("ner_embeds: %s" % " ".join([str(x) for x in ner_embed]))
        
        # combine all features
        features.append(
            InputFeatures(input_ids=input_ids, # see InputFeatures class above on line 44
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          intent_label_id=intent_label_id,
                          slot_labels_ids=slot_labels_ids,
                          ner_embed=ner_embed
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    # initiate JointProcessor(args) 
    processor = processors[args.task](args)
    
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format( 
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file):
        # Load data features from cache
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        # features is a dictionary including 'attention_mask', 'input_ids', 'intent_label_id' 
        # 'slot_labels_ids', 'token_type_ids'
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train") # see get_examples() function on line 111
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")
                
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index # use 0 as padding
        # convert examples above into set of features, a dictionary including 
        # 'attention_mask', 'input_ids', 'intent_label_id', 'slot_labels_ids', 'token_type_ids'
        features = convert_examples_to_features(mode, args, examples, args.max_seq_len, tokenizer,
                                                pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file) # save the features into cached_features_file

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)
    all_ner_embeds = torch.tensor([f.ner_embed for f in features], dtype=torch.float)
    
    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_intent_label_ids, all_slot_labels_ids,
                            all_ner_embeds)
    
    return dataset
