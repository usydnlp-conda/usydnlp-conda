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
$ python3 main.py --task {task_name} \
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

# for APP
set FLASK_ENV=development
python app.py

# For testing code
python main.py --task conda --model_type bert --model_dir conda_testing_model --do_train --num_train_epochs 1

python predict.py --input_file sample_pred_in.txt --output_file sample_pred_out.txt --model_dir conda_testing_model

# Processing data
vocab_process.py

## Prediction
$ python predict.py --input_file {INPUT_FILE_PATH} --output_file {OUTPUT_FILE_PATH} --model_dir {SAVED_CKPT_PATH}

# Example
python predict.py --input_file sample_pred_in.txt --output_file sample_pred_out.txt --model_dir final_conda_bert_dg_model

## Updates

- 2019/12/03: Add DistilBert and RoBERTa result
- 2019/12/14: Add Albert (large v1) result
- 2019/12/22: Available to predict sentences
- 2019/12/26: Add Albert (xxlarge v1) result
- 2019/12/29: Add CRF option
- 2019/12/30: Available to check `sentence-level semantic frame accuracy`
- 2020/01/23: Only show the result related with uncased model
- 2020/04/03: Update with new prediction code

## References

- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [pytorch-crf](https://github.com/kmkurn/pytorch-crf)
