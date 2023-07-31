## PaperPersiChat

This repo contains code for the SigDial 2023 paper: "PaperPersiChat: Scientific Paper Discussion Chatbot using Transformers and Discourse Flow Management"


**Description:**

PaperPersiChat is a chatbot-system designed for the discussion of scientific papers. This system supports summarization and question-answering modes within a single end-to-end chatbot pipeline, which is guided by discourse analysis. To expedite the development of similar systems, we also release the gathered dataset, which has no publicly available analogues.

PaperPersiChat is running online on http://www.PaperPersiChat.tech



### Installation

Works only for the Python 3.8 environment.

1) Install requirements line by line for the needed cuda:

```bash
cat requirements.txt | xargs -n 1 pip install --extra-index-url https://download.pytorch.org/whl/cu111
```

2) Download checkpoints:

```bash
git lfs install
git clone https://huggingface.co/ai-forever/paper_persi_chat checkpoint
```

3) Copy data to the required folders:

```bash
cp checkpoint/glove.6B.100d.txt chat_scripts/dialogue_discourse_parser/glove/ && cp -r checkpoint/convokit_50_model chat_scripts/dialogue_discourse_parser/convokit_50_model && cp checkpoint/convokit_dials_train.json chat_scripts/dialogue_discourse_parser/data
```


***Note***: ```papers_segmented_data``` contains the subset of papers used for the official demo and can be extended manually in the same format. File ```titles_all.pkl``` consists  of the list of titles in jsons, and ```words2papers_all.pkl``` consists of the dictionary whose keys are words and values are sets of title indices from the titles_all list.


### How to use

Run demo app on localhost:XXXX using     

```bash
cd chat_scripts && streamlit run streamlit_demo.py --server.port=XXXX
```

### Dataset

Dataset is available via https://huggingface.co/datasets/ai-forever/paper_persi_chat

### Training scripts

Most of the training scripts are available in the ```training_scripts``` folder:

- ```Dialogue Dataset Generation``` notebooks were used to collect datasets using OpenAI API.

- ```Bart Finetuning - summary.ipynb``` was used to train the Summarizer module.

- ```DebertaQA``` notebooks were used to construct datasets for the QA module. 

- ```BART Response Generation.ipynb``` was used to train the Response Generator module module.

- ```DialogueDiscourseParser``` and ```Relation Classification``` were used to train the DM module.
