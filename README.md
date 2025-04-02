## DS 4300 - Practical 02 – RAG System
#### Final Deliverables Due: Wednesday, April 2 @ 11:59pm
~~~
Team Members:
- Eva Balogun
- Olivia Mintz
- Nusha Bhat

GitHub Link: https://github.com/ebalo03/P2-Vector-DBs-LLMs
Slides Link: https://docs.google.com/presentation/d/1aOePmcNDtWSWX47PMSNGDYFN1ik1AcfmukTtWA5UAho/edit?usp=sharing
~~~

## Overview

This practical implements a local Retrieval-Augmented Generation (RAG) system that enables users to query DS 4300 course notes. The pipeline:

- Ingests team-collected notes (PDFs, slides)
- Chunks and embeds the text
- Indexes embeddings into a vector database
- Accepts user queries and retrieves relevant context
- Passes context to a local LLM via Ollama to generate a response

## Setup Instructions

- Clone Repo
git clone https://github.com/ebalo03/P2-Vector-DBs-LLMs.git
cd P2-Vector-DBs-LLMs
- Activate Virtual Environment
- Install Dependencies
pip install -r requirements.txt
- Install & Run Ollama
- Pull Models used for this practical
ollama pull mistral
ollama pull nous-hermes
- Run Pipeline
chunk size, overlap, embedding-model,  vector-db, llm, query


## Evaluation
We evaluated our pipeline using different chunks/overlap sizes, 3 embedding models, 3 vector dbs, 3 local LLMS, query variations. Our results are described in our slide deck linked above.

