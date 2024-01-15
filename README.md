# usc_isi_reu
Summer Research Data Analysis Project at USC Information Science Institute

Distribution of % of Emotional Language Across Left, Center, and Right biased news sources:
<img width="693" alt="infographic" src="https://github.com/chen70335/usc_isi_reu/assets/101837218/0358ccfd-f909-4a31-8e7a-f0bce679a09d">

## datasets/sentenced/
Collection of Datasets Generated from POLUSA's political articles dataset (https://zenodo.org/record/3813664)
Created by running dataset_curaiton.py on each news source
Used for gpt_api.py and data visualization figures

## dataset_curation.py
Utilizes Google Natural Language API and Huggingface pre-trained Transformer-based models
Utilizes pandas to collect and curate dataset

## gpt_api.py
Utilizes openAI GPT API to rephrase sentences and create randomly shuffled annotation files for research purposes

## figure1, figure2, figure3
Utilizes pandas, numpy, matplotlib to generate data visualizations such as boxplots, histograms, and stacked bar graphs

