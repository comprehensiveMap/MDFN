Codes accompanying the paper [Filling the Gap of Utterance-aware and Speaker-aware Representation for Multi-turn Dialogue](https://arxiv.org/pdf/2009.06504.pdf)

## Instruction
Our code is compatible with compatible with python 3.x so for all commands listed below python is `python3`.

We strongly suggest you to use `conda` to control the virtual environment.

- Install requirements

``
pip install -r requirements.txt
``

- Train the model and predict.

```
python run_MDFN.py \
--data_dir datasets/mutual \
--model_name_or_path \
google/electra-large-discriminator \
--model_type electra \
--task_name mutual\
--output_dir output_mutual_electra \
--cache_dir cached_models \
--max_seq_length 256 \
--do_train --do_eval \
--train_batch_size 6 \
--eval_batch_size 6 \
--learning_rate 4e-6 \
--num_train_epochs 3 \
--gradient_accumulation_steps 1 \
--local_rank -1 \
```

## Reference
If you use this code please cite our paper:

```
@inproceedings{liu2021filling,
  title={Filling the Gap of Utterance-aware and Speaker-aware Representation for Multi-turn Dialogue},
  author={Liu, Longxiang and Zhang, Zhuosheng and and Zhao, Hai and Zhou, Xi and Zhou, Xiang},
  booktitle={The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21)},
  year={2021}
}
```