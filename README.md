# usc_isi_reu
Summer Research Data Analysis Project at USC Information Science Institute

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
