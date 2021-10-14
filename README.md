# MAC, BERT & DAD: Multi-Aspect Cross-Attention, Joint-Bert and Dual-Annotated Datasets for In-Game Toxicity Detection

## Sony Jufri

<b>Abstract:</b> In the past few years, the popularity of online multi-player gaming communities had grown rapidly. Unfortunately, with this phenomenon, came along the emergence of harmful behaviour in these communities. Many attempts had been done by gaming companies and Natural Language Processing (NLP) researchers to detect online abuse. These attempts ranged from the simple use of human reviewers to the more sophisticated machine learning and deep learning techniques. Nevertheless, recent studies showed that online abuse remained a big issue for many people. It is important that we remain vigilant and continue to improve our effort in detecting online abuse. 

Inspired by two of the most recent literatures in abusive language detection from the University of Sydney's NLP Group, we developed some deep learning models that could better detect toxic behaviour in online gaming. Using data from two well-known games, Defense of the Ancients 2 (Dota 2) and League of Legends (LOL), a variety of Joint-BERT models would be used and further improved to better detect toxic behaviour in these games.

## JointBERT

The Base Joint-BERT model used the (Unofficial) Pytorch implementation of `JointBERT`: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)

## The Base Joint-BERT Model Architecture

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68875755-b2f92900-0746-11ea-8819-401d60e4185f.png" />  
</p>

- Predict `intent` and `slot` at the same time from **one BERT model** (=Joint model)
- total_loss = intent_loss + coef \* slot_loss (Change coef with `--slot_loss_coef` option)
- **If you want to use CRF layer, give `--use_crf` option**

## Dependencies

- python>=3.6
- torch==1.8.1
- transformers==3.0.2
- seqeval==0.0.12
- pytorch-crf==0.7.2

## Dataset

|       | Train  | Dev   | Test  | Intent Labels | Slot Labels |
| ----- | ------ | ----- | ----- | ------------- | ----------- |
| CONDA | 26,078 | 8,705 | 0     | 4             | 6           |
| LOL   | 29,358 | 3,258 | 3,628 | 4             | 6           |

- The number of labels are based on the _train_ dataset.
- Add `UNK` for labels (For intent and slot labels which are only shown in _dev_ and _test_ dataset)
- Add `PAD` for slot label

## Training & Evaluation
$ python main.py --task {task_name} \
                 --model_type {model_type} \
                 --model_dir {model_dir_name} \
                 --do_train --do_eval \
                 --add_embed {d,g,e,dg,de} \
		 --optim_metric {JSA, UCA, U-F1(I), T-F1, T-F1(T), etc}

### Examples
### CONDA
python main.py --task conda --model_type bert --model_dir final_conda_bert_dg_model --do_train --add_embed dg --optim_metric JSA

python main.py --task conda --model_type bert --model_dir final_conda_bert_g_model --do_train --add_embed g --optim_metric JSA

python main.py --task conda --model_type bert --model_dir final_conda_bert_E_model --do_train --add_embed E --optim_metric JSA

### LOL
python main.py --task low --model_type bert --model_dir final_low_bert_dg_model --do_train --do_eval --add_embed dg --optim_metric JSA

python main.py --task low --model_type distilbert --model_dir final_low_distilbert_e_model --do_train --do_eval --add_embed e --optim_metric U-F1(I)

python main.py --task low --model_type distilbert --model_dir final_low_distilbert_de_model --do_train --do_eval --add_embed de --optim_metric U-F1(I)

## Processing data
vocab_process.py

## Prediction
$ python predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}

### Example
python predict.py --input_file sample_pred_in.txt --output_file sample_pred_out.txt --model_dir final_conda_bert_dg_model

## Deployment using Flask API
set FLASK_ENV=development  \
python app.py

## References

- [monologg/JointBERT](https://github.com/monologg/JointBERT)
- [avinassh/pytorch-flask-api-heroku](https://github.com/avinassh/pytorch-flask-api-heroku)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)

<b>Note:</b> the base code for the Joint-BERT models were sourced from the [monologg/JointBERT](https://github.com/monologg/JointBERT) GitHub page and the deployment codes were adopted from [avinassh/pytorch-flask-api-heroku](https://github.com/avinassh/pytorch-flask-api-heroku) GitHub website.