# Improving Readability Assessment with Ordinal Log-Loss
19th Workshop on Innovative Use of NLP for Building Educational Applications

## Abstract
Automatic Readability Assessment (ARA) predicts the level of difficulty of a text, e.g. at Grade 1 to Grade 12. ARA is an ordinal classification task since the predicted levels follow an underlying order, from easy to difficult. However, most neural ARA models ignore the distance between the gold level and predicted level, treating all levels as independent labels. This paper investigates whether distance-sensitive loss functions can improve ARA performance. We evaluate a variety of loss functions on neural ARA models, and show that ordinal log-loss can produce statistically significant improvement over the standard cross-entropy loss in terms of adjacent accuracy in a majority of our datasets.

## Tools
scikit-learn==0.24.1<br>
torch==1.11.0<br>
transformers==4.5.0<br>

## Pretrained Model
MacBERT: https://huggingface.co/hfl/chinese-macbert-large<br>
BERT: https://huggingface.co/bert-base-uncased<br>
BART: https://huggingface.co/facebook/bart-base<br>
RoBERTa: https://huggingface.co/FacebookAI/roberta-base<br>
XLNet: https://huggingface.co/xlnet/xlnet-base-cased<br>

For English experiments, we use the learning rate of 2e-5 for BERT and 3e-5 for the other pre-trained language models.<br>
For Chinese experiments, we use the learning rate of 2e-5 for MacBERT.

# How to Run
Most of the code is based on https://github.com/yjang43/pushingonreadability_transformers

1. Copy and go to pushingonreadability_transformers-master folder

2. 5-Fold datasets in data folder.
- Stratified folds of data will save under file name _"data/onestop.{k}.{type}.csv"_.
_k_ means _k_-th of the K-Fold and _type_ is either train, valid, or test.


3. Fine-tune on dataset with pretrained model using train{corpus-name}{loss-type}.py file from "code" folder.
```bash
python trainOnestopOLL1.py --corpus_name onestop --model bert --learning_rate 2e-5
```

4. Collect output probability with a trained model.

```bash
python inference.py --checkpoint_path checkpoint/onestop.bert.0.14 --data_path data/onestop.0.test.csv
```

5. Collect output probability for each grade.

6. Use NerualModelFullFoldResult.ipynb to evaluate the result

## References
Pushing on Text Readability Assessment: A Transformer Meets Handcrafted Linguistic Features<br>
