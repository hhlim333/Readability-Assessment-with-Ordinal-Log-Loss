# Readability-Assessment-with-Ordinal-Log-Loss
19th Workshop on Innovative Use of NLP for Building Educational Applications

## Abstract
Automatic Readability Assessment (ARA) predicts the level of difficulty of a text, e.g. at Grade 1 to Grade 12. ARA is an ordinal classification task since the predicted levels follow an underlying order, from easy to difficult. However, most neural ARA models ignore the distance between the gold level and predicted level, treating all levels as independent labels. This paper investigates whether distance-sensitive loss functions can improve ARA performance. We evaluate a variety of loss functions on neural ARA models, and show that ordinal log-loss can produce statistically significant improvement over the standard cross-entropy loss in terms of adjacent accuracy in a majority of our datasets.

## Tools
scikit-learn==0.24.1<br>
torch==1.11.0<br>
transformers==4.5.0<br>

## Pretrained Model
MacBERT: https://huggingface.co/hfl/chinese-macbert-large<br>
BERT: https://huggingface.co/bert-base-chinese<br>
BERT-wwm: https://huggingface.co/hfl/chinese-bert-wwm<br>
RoBERTa: https://huggingface.co/hfl/chinese-roberta-wwm-ext<br>

We use the learning rate of 2e-5 for all pretrained Model

# How to Run
Most of the code is based on https://github.com/yjang43/pushingonreadability_transformers

1. Go to pushingonreadability_transformers-master folder

2. Create 5-Fold of a dataset for training.
```bash
python kfold.py --corpus_path mainland.csv --corpus_name mainland
```
- Stratified folds of data will save under file name _"data/mainland.{k}.{type}.csv"_.
_k_ means _k_-th of the K-Fold and _type_ is either train, valid, or test.


3. Fine-tune on dataset with pretrained model.
```bash
python train.py --corpus_name mainland --model chinese-macbert-large --learning_rate 2e-5
```

4. Collect output probability with a trained model.

```bash
python inference.py --checkpoint_path checkpoint/mainland.chinese-macbert-large.0.14 --data_path data/mainland.0.test.csv
```

5. Collect output probability for each grade.

## References
Pushing on Text Readability Assessment: A Transformer Meets Handcrafted Linguistic Features<br>
