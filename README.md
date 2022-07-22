---
language: en
license: apache-2.0
datasets:
- sst2
- glue
model-index:
- name: distilbert-base-uncased-finetuned-sst-2-english
  results:
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: glue
      type: glue
      config: sst2
      split: validation
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9105504587155964
      verified: true
    - name: Precision
      type: precision
      value: 0.8978260869565218
      verified: true
    - name: Recall
      type: recall
      value: 0.9301801801801802
      verified: true
    - name: AUC
      type: auc
      value: 0.9716626673402374
      verified: true
    - name: F1
      type: f1
      value: 0.9137168141592922
      verified: true
    - name: loss
      type: loss
      value: 0.39013850688934326
      verified: true
  - task:
      type: text-classification
      name: Text Classification
    dataset:
      name: sst2
      type: sst2
      config: default
      split: train
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9885521685548412
      verified: true
    - name: Precision Macro
      type: precision
      value: 0.9881965062029833
      verified: true
    - name: Precision Micro
      type: precision
      value: 0.9885521685548412
      verified: true
    - name: Precision Weighted
      type: precision
      value: 0.9885639626373408
      verified: true
    - name: Recall Macro
      type: recall
      value: 0.9886145346602994
      verified: true
    - name: Recall Micro
      type: recall
      value: 0.9885521685548412
      verified: true
    - name: Recall Weighted
      type: recall
      value: 0.9885521685548412
      verified: true
    - name: F1 Macro
      type: f1
      value: 0.9884019815052447
      verified: true
    - name: F1 Micro
      type: f1
      value: 0.9885521685548412
      verified: true
    - name: F1 Weighted
      type: f1
      value: 0.9885546181087554
      verified: true
    - name: loss
      type: loss
      value: 0.040652573108673096
      verified: true
---

# DistilBERT base uncased finetuned SST-2

This model is a fine-tune checkpoint of [DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased), fine-tuned on SST-2.
This model reaches an accuracy of 91.3 on the dev set (for comparison, Bert bert-base-uncased version reaches an accuracy of 92.7).

For more details about DistilBERT, we encourage users to check out [this model card](https://huggingface.co/distilbert-base-uncased).

# Fine-tuning hyper-parameters

- learning_rate = 1e-5
- batch_size = 32
- warmup = 600
- max_seq_length = 128
- num_train_epochs = 3.0

# Bias

Based on a few experimentations, we observed that this model could produce biased predictions that target underrepresented populations.

For instance, for sentences like `This film was filmed in COUNTRY`, this binary classification model will give radically different probabilities for the positive label depending on the country (0.89 if the country is France, but 0.08 if the country is Afghanistan) when nothing in the input indicates such a strong semantic shift. In this [colab](https://colab.research.google.com/gist/ageron/fb2f64fb145b4bc7c49efc97e5f114d3/biasmap.ipynb), [Aurélien Géron](https://twitter.com/aureliengeron) made an interesting map plotting these probabilities for each country.

<img src="https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/map.jpeg" alt="Map of positive probabilities per country." width="500"/>

We strongly advise users to thoroughly probe these aspects on their use-cases in order to evaluate the risks of this model. We recommend looking at the following bias evaluation datasets as a place to start: [WinoBias](https://huggingface.co/datasets/wino_bias), [WinoGender](https://huggingface.co/datasets/super_glue), [Stereoset](https://huggingface.co/datasets/stereoset).
