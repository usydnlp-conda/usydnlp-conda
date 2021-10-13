# MAC, BERT & DAD: Multi-Aspect Cross-Attention, Joint-Bert and Dual-Annotated Datasets for In-Game Toxicity Detection

### Sony Jufri

<b>Abstract:</b> In the past few years, the popularity of online social media and multi-player gaming communities had grown rapidly. Unfortunately, with this phenomenon, came along the emergence of harmful behaviour in these communities. A study by Amnesty International (2017) found that 7.1% of tweets (which amounted to 1.1 million tweets or one every 30 seconds) that were sent to the women in the study, were abusive. Women of colour in particular, were the more likely target (34%). Another study by the Australian Government’s eSafety Commissioner (2021) reported that 17% of young Australians (i.e., more than 200,000 people) that played online games, had experienced some forms of bullying or abuse. 

Many attempts had been done by social media companies and Natural Language Processing (NLP) researchers to detect online abuse. These attempts ranged from the simple use of human reviewers to the more sophisticated machine learning and deep learning techniques. Nevertheless, the above studies showed how online abuse remained a big issue for many people to this day. Therefore, it was important that we remained vigilant and continued to improve our effort in detecting online abuse. 

Inspired by the methods discussed in two of the most recent literatures in abusive language detection from the University of Sydney's NLP Group, we aimed to develop a deep learning model that could better detect toxic behaviour in online gaming. Using data from two well-known games, Defense of the Ancients 2 (Dota 2) and League of Legends (LOL), a variety of Joint-BERT models would be used and further improved to better detect toxic behaviour in these games. We hoped that this study would benefit not only online gamers out there, but also other stakeholders in the gaming industry.

# JointBERT

(Unofficial) Pytorch implementation of `JointBERT`: [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)

## Model Architecture

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/68875755-b2f92900-0746-11ea-8819-401d60e4185f.png" />  
</p>

- Predict `intent` and `slot` at the same time from **one BERT model** (=Joint model)
- total_loss = intent_loss + coef \* slot_loss (Change coef with `--slot_loss_coef` option)
- **If you want to use CRF layer, give `--use_crf` option**

## Dependencies

- python>=3.6
- torch==1.6.0
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

```bash
$ python main.py --task {task_name} \
                  --model_type {model_type} \
                  --model_dir {model_dir_name} \
                  --do_train --do_eval \
                  --add_embed {d,g,e,dg,de} \
		  --optim_metric {JSA, UCA, U-F1(I), T-F1, T-F1(T), etc}

# Examples
# for CONDA
python main.py --task conda --model_type bert --model_dir final_conda_bert_dg_model --do_train --add_embed dg --optim_metric JSA

python main.py --task conda --model_type bert --model_dir final_conda_bert_g_model --do_train --add_embed g --optim_metric JSA

python main.py --task conda --model_type bert --model_dir final_conda_bert_E_model --do_train --add_embed E --optim_metric JSA

# for LOL
python main.py --task low --model_type bert --model_dir final_low_bert_dg_model --do_train --do_eval --add_embed dg --optim_metric JSA

python main.py --task low --model_type distilbert --model_dir final_low_distilbert_e_model --do_train --do_eval --add_embed e --optim_metric U-F1(I)

python main.py --task low --model_type distilbert --model_dir final_low_distilbert_de_model --do_train --do_eval --add_embed de --optim_metric U-F1(I)

# For testing code
python main.py --task conda --model_type bert --model_dir conda_testing_model --do_train --num_train_epochs 1

python predict.py --input_file sample_pred_in.txt --output_file sample_pred_out.txt --model_dir conda_testing_model

# Processing data
vocab_process.py

## Prediction
$ python predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}

# Example
python predict.py --input_file sample_pred_in.txt --output_file sample_pred_out.txt --model_dir final_conda_bert_dg_model

# for APP
set FLASK_ENV=development
python app.py

## References

- The base code for the Joint-BERT models were sourced from the monologg GitHub page: https://github.com/monologg/JointBERT
- The app was developed using PyTorch Flask API. The code was adopted from avinassh GitHub website: https://github.com/avinassh/pytorch-flask-api-heroku
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)