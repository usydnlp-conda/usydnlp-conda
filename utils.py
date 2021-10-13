import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

from transformers import BertConfig, DistilBertConfig, AlbertConfig
from transformers import BertTokenizer, DistilBertTokenizer, AlbertTokenizer

from model import JointBERT, JointDistilBERT, JointAlbert
from sklearn.metrics import classification_report

MODEL_CLASSES = {
    'bert': (BertConfig, JointBERT, BertTokenizer),
    'distilbert': (DistilBertConfig, JointDistilBERT, DistilBertTokenizer),
    'albert': (AlbertConfig, JointAlbert, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-xxlarge-v1'
}


def get_intent_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]


def get_slot_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):
    # load tokenizer based on selected bert model (bert, distilbert, albert)
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path) 


def init_logger():
    # configures the root logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

   
def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels, args):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels) # calculate intent accuracy 
    slot_result = get_slot_metrics(slot_preds, slot_labels) # calculate P, R and F1 scores for slots 
    semantic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels) # calculate sentence frame accuracy 
    
    results.update(intent_result)
    results.update(slot_result)
    results.update(semantic_result)
    
    # obtain intent and slot f1 for conda and low datasets
    if args.task=='low':
        intent_f1 = get_intent_f1(intent_preds, intent_labels) # calculate intent f1
        results.update(intent_f1)
        slot_f1 = get_slot_f1_low(slot_preds, slot_labels)
        results.update(slot_f1)
    
    if args.task=='conda':
        intent_f1 = get_intent_f1(intent_preds, intent_labels) # calculate intent f1
        results.update(intent_f1)
        slot_f1 = get_slot_f1_conda(slot_preds, slot_labels)
        results.update(slot_f1)
    
    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }


def get_intent_f1(preds, labels):
    f1_intent = classification_report(preds, labels, digits=4, output_dict=True)
    f1_a = f1_intent['1']['f1-score']
    f1_e = f1_intent['2']['f1-score']
    f1_i = f1_intent['3']['f1-score']
    f1_o = f1_intent['4']['f1-score']
    return {
        "U-F1(A)": f1_a,
        "U-F1(E)": f1_e,
        "U-F1(I)": f1_i,
        "U-F1(O)": f1_o
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    semantic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "semantic_frame_acc": semantic_acc
    }


def get_slot_f1_low(slot_preds, slot_labels):
    """For extracting slot-f1 metrics for low dataset"""
    slot_preds = [item for sublist in slot_preds for item in sublist]
    slot_labels = [item for sublist in slot_labels for item in sublist]
    f1_slot = classification_report(slot_preds, slot_labels, digits=4, output_dict=True)
    f1_t = classification_report(slot_preds, slot_labels, digits=4, output_dict=True, labels=['C','L','P','S','T'])
    
    t_f1 = f1_t['micro avg']['f1-score']
    f1_c = f1_slot['C']['f1-score']
    f1_l = f1_slot['L']['f1-score']
    f1_o = f1_slot['O']['f1-score']
    f1_p = f1_slot['P']['f1-score']
    f1_s = f1_slot['S']['f1-score']
    f1_t = f1_slot['T']['f1-score']
    return {
        "T-F1": t_f1,
        "T-F1(C)": f1_c,
        "T-F1(L)": f1_l,
        "T-F1(O)": f1_o,
        "T-F1(P)": f1_p,
        "T-F1(S)": f1_s,
        "T-F1(T)": f1_t
    }

def get_slot_f1_conda(slot_preds, slot_labels):
    """For extracting slot-f1 metrics for conda dataset"""
    slot_preds = [item for sublist in slot_preds for item in sublist]
    slot_labels = [item for sublist in slot_labels for item in sublist]
    f1_slot = classification_report(slot_preds, slot_labels, digits=4, output_dict=True)
    f1_t = classification_report(slot_preds, slot_labels, digits=4, output_dict=True, labels=['C','D','P','S','T'])
    
    t_f1 = f1_t['micro avg']['f1-score']
    f1_c = f1_slot['C']['f1-score']
    f1_d = f1_slot['D']['f1-score']
    f1_o = f1_slot['O']['f1-score']
    f1_p = f1_slot['P']['f1-score']
    f1_s = f1_slot['S']['f1-score']
    f1_t = f1_slot['T']['f1-score']
    return {
        "T-F1": t_f1,
        "T-F1(C)": f1_c,
        "T-F1(D)": f1_d,
        "T-F1(O)": f1_o,
        "T-F1(P)": f1_p,
        "T-F1(S)": f1_s,
        "T-F1(T)": f1_t
    }