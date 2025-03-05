---
hide:
#   - navigation
  - toc
---

# T5-Base, T5-Large, and BART — The Battle of the Summarization Models

![transformers](img/transformers.png)

## Project Overview
This project explores text summarization _transformer models_ T5-Base, T5-Large and BART-CNN and compare their _summarization_ results. Building upon the Reddit NYCApartment dataset that was pulled using the API PRAW, the focus is to evaluate and compare these models when summarizing user comments and posts.
By experimenting with the pipeline API, fine-tuning parameters, and analyzing different model sizes, we gain insights into the strengths and weaknesses of each approach for summarization tasks.

✽ The full article can be found on my page on [medium.com](https://medium.com/@gabya06/t5-vs-bart-the-battle-of-the-summarization-models-c1e6d37e56ca).<br>
✽ The corresponding code can be found in [github](https://github.com/Gabya06/RentWatchAI/blob/main/post_summarization.ipynb).<br>


## Table of Contents

1. [Data Visualization & Analysis](#data-viz)
2. [Pipelines for Summarization](#pipelines-for-summarization)
    <br>a. [T5-BASE Model](#t5-base)
    <br>b. [T5-LARGE Model](#t5-large)
    <br>c. [BART-LARGE-CNN Model](#bart)
3. [More Parameter Control](#more-control)
4. [Exploring `num_beans` Parameter](#num_beans)
5. [Summarizing the Full Dataset](#summarization)

<a name="data-viz"></a>
### 1. Data Visualization & Analysis

Before diving into the summarization models, I first analyzed the Reddit dataset, which contains posts and comments from the r/NYCApartment subreddit. I used histograms to analyze post sentiment distributions and analyzed the top 25% most engaging posts. This helped inform how summarization could be applied effectively.


<a name="pipelines-for-summarization"></a>
### 2. Pipelines for Summarization

For text summarization, I utilized Hugging Face’s pipeline API, which simplifies the process of using transformer models. Below is an example of how the summarization pipeline is used:

``` py linenums="1"
# summarize just one comment using t5-base
summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base")
summary = summarizer(sample_comment, min_length=5, max_length=200,
                     do_sample=False)
```

Each model has its unique characteristics and produces different summary results. Below is a comparison of the models tested.


<a name="t5-base"></a>
### a. T5-BASE Model 
    * **Model size**: 220M parameters
    * **Strengths**: Balanced performance and efficiency
    * **Weaknesses**: Less fluent and less coherent than larger models
    * **Best for**: General text summarization, Q&A, and text generation

<a name="t5-large"></a>
### b. T5-LARGE Model
    * **Model size**: 770M parameters
    * **Strengths**: More fluent and detailed summaries
    * **Weaknesses**: Requires more computational power
    * **Best for**: Complex summarization tasks, longer documents


<a name="bart"></a>
### c. BART-LARGE-CNN Model

```py linenums="1"
# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load the tokenizer separately to enable manual truncation
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

summary = summarizer(sample_comment, min_length=30, max_length=100,
                     do_sample=False)
```

* **Model size**: 406M parameters
* **Strengths**: High-quality abstractive summaries
* **Weaknesses**: Requires fine-tuning for domain-specific texts
* **Best for**: News summarization, content condensation

### T5 vs. BART Comparison:
| Feature| T5	|BART|
---------|-------|-----|
| Model Type	| Encoder-Decoder |	Encoder-Decoder | 
| Pretraining	| Text-to-Text	| Denoising Autoencoder | 
| Output Style	| Keeps key phrases	| Rephrases more | 
| Performance	| Works well for structured text	| Handles noisy text better | 
| Ideal Use Case	| Summarizing clean, factual text	| Summarizing complex or opinion-heavy content| 


<a name="more-control"></a>
### 3. More Parameter Control

While the `pipeline` method is very easy and convenient, manually loading the model lets us to better customize summarization results. The `generate` function allows for additional control over summarization.

``` py linenums="1"
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
summary_ids = model.generate(**inputs, max_length=60, min_length=30, do_sample=False)
summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
```


<a name="num_beans"></a>
### 4. Exploring `num_beans` Parameter

`num_beams` controls the number of possible candidate sequences considered during beam search decoding. Setting `num_beams=1` is the equivalent to _greedy search_ which might be _fastest_ but does not always provide the most coherent result. Higher values improve quality but increase computation time.

|`num_beams` | Summary Quality | Computational Time |
|------------|------------|------------|
| 1 (greedy) | Lowest | Fastest |
| 4 | Moderate | Medium |
8 | High | Slower|


<a name="summarization"></a>
### 5. Summarizing the Full Dataset

Once the individual summarization tests were complete, I applied the summarization pipeline to the entire dataset to generate summaries for all Reddit comments. This step is important for handling large datasets efficiently.


``` py linenums="1"
import pandas as pd

# Apply summarization to each post's comment
df['summary'] = df['comments'].apply(lambda x: summarizer(x, max_length=60, min_length=30, do_sample=False)[0]['summary_text'])
```

### Final Summary of Model Performance

| Model | Size | Strengths | Weaknesses | 
| ------ |------|------|------|
|`T5-BASE` | 220M | Good performance & Efficient | Less coherent summaries |
`T5-LARGE` |770M | More detailed output | Higher computational cost |
`BART-CNN-LARGE`| 406M | Higher quality summaries | Requires fine-tuning for domain specific tasks |


This project demonstrates how different transformer models can be leveraged for text summarization, showcasing various methods for customizing summarization quality based on the specific needs of a dataset.

---
### Resources
* [Hugging Face](https://huggingface.co/models?pipeline_tag=summarization&p=1&sort=trending&search=t5)
* [Beam Search](https://towardsdatascience.com/foundations-of-nlp-explained-visually-beam-search-how-it-works-1586b9849a24/)
