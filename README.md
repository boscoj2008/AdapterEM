# AdapterEM: Pre-trained Language Model Adaptation fo Generalized Entity Matching using Adapter-tuning

abstract 

Entity Matching (EM) involves identifying different data representations referring to the same entity from multiple data sources and is typically formulated as a binary classification problem. It is a challenging problem in data integration due to the heterogeneity of data representations. State-of-the-art solutions have adopted NLP techniques based on word embedding technology and pre-trained language models (PrLMs) via the fine-tuning paradigm yielding successful results in EM, however, sequential fine-tuning of overparameterized PrLMs can require costly parameter updates resulting in catastrophic forgetting, especially in low-resource scenarios. Our study proposes a parameter-efficient paradigm for fine-tuning PrLMs based on adapters, small neural networks introduced between the layers of a PrLM to mitigate problems introduced by regular fine-tuning, where only the adapter weights are updated via backpropagation while the PrLM parameters are frozen. Adapter-based methods have been successfully applied to multilingual speech problems achieving promising results, however, the effectiveness of these methods when applied to EM is not yet well understood, in particular for generalized EM with heterogeneous data. We show that by utilizing less than 13\% of the parameters with a significantly smaller computational footprint, our solution mitigates catastrophic forgetting and matches or outperforms regular fine-tuning and prompt-tuning baselines.

Paper [AdapterEM: Pre-trained Language Model Adaptation fo Generalized Entity Matching using Adapter-tuning](link).

![Adapter fine-tuning. ](./imgs/adapters.png)

## Requirements

- Python 3.9.7
- torch 1.12.1+cu113 
- torchvision 0.13.1+cu113
- Adapter Transformers 4.24.0
- scikit-learn 1.1.2

 run 
```
python install -r requirements.txt
```

## Datasets

We use eight real-world benchmark datasets with different structures from [Machamp](https://github.com/megagonlabs/machamp) and [Geo-ER](https://github.com/PasqualeTurin/Geo-ER) also used in the 
[PromptEM paper](https://arxiv.org/abs/2207.04802).

## Fine-tuning

To train and evaluate with AdapterEM with a randomly initialized adapter;

```
bash semi-homo.sh
```

To train a invertible language adapter using mask language modeling (probability p=0.20), navigate to `masked_language_modeling`.
After training, language adapter is saved to `adapters` directory, then run;

```
bash semi-homo_tapt-20.sh
```
or if langauge adapter trained for p=40\% then,


```
bash semi-homo_tapt-40.sh
```



The meaning of the flags:

- `--model_name_or_path`: the name or local path of the pre-trained language model. e.g. `bert-base-uncased`
- `--data_name`: the name of the dataset. options: `[rel-heter, rel-text, semi-heter, semi-homo, semi-rel, semi-text-c,semi-text-w, geo-heter, all]`
- `--adapter_setup`: specify the configuration i.e., blank adapter or pre-trained e.g task_only
- `--adapter_size`: bottleneck size e.g `2, 4,..`
- `--k`: the proportion of training data used. e.g. `0.1 or 1.0`
- `--num_iter`: the number of iterations. e.g. `1`
- `--text_summarize`: the flag to enable text summarization in entity serialization.
- `--add_token`: the flag to add special token in entity serialization.
- `--max_length`:  the maximum length (in number of tokens) for the inputs to the transformer model. e.g. `512`
- `--n`: the number of epochs for model training. e.g. `30`
- `--batch_size`: batch size. e.g. `32`
- `--lr`: learning rate. e.g.`1e-4`

## Training snli language adapter

```
bash train_slni_adapter.sh

```   


## License

