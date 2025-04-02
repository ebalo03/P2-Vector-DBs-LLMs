## Practical 02 – Vector DBs & LLMs
### Project needs to be functional by March 24’s exam
### Final deliverables are due WED APRIL 2 @ 11:59pm
### More info forthcoming on deliverables.
~~~
Team Members: Eva Balogun, Olivia Mintz, Nusha Bhat
GitHub Link: https://github.com/ebalo03/P2-Vector-DBs-LLMs
Slides Link: https://docs.google.com/presentation/d/1aOePmcNDtWSWX47PMSNGDYFN1ik1AcfmukTtWA5UAho/edit?usp=sharing
~~~


Overview:

In this project, you and your team will build a local Retrieval-Augmented Generation system that allows a user to query the collective DS4300 notes from members of your team.  Your system will do the following:

Ingest a collection of documents that represent material, such as course notes, you and your team have collected throughout the semester. 
Index those documents using embedding and a vector database
Accept a query from the user. 
Retrieve relevant context based on the user’s query
Package the relevant context up into a prompt that is passed to a locally-running LLM to generate a response. 

In this, you’ll experiment with different variables - chunking strategies, embedding models, some prompt engineering, local LLM, and vector database options.  You’ll analyze how these changes affect the system’s performance and output quality. 

Tools:

Python for building the pipeline
Ollama for running LLMs locally (you’ll compare and contrast at least 2 different models
Vector Databases (Redis Vector DB, Chroma, and one other of your choosing)
Embedding Models (you’ll compare and contrast at least 3 different options)  

Variables to Explore:

Text preprocessing & chunking 
Try various size chunks: 200, 500, 1000 tokens, for example
Try different chunk overlap sizes: 0, 50, 100 token overlap for example
Try various basic text prep strategies such as removing whitespace, punctuation, and any other “noise”. 
Embedding Models - Choose 3 to compare and contrast. Examples include: 
sentence-transformers/all-MiniLM-L6-v2
sentence-transformers/all-mpnet-base-v2
InstructorXL
The model we used in class
Measure interesting properties of using the various embedding models such as speed, memory usage, and retrieval quality (qualitative). 
Vector Database - At a minimum, compare and contrast Redis Vector DB and Chroma (you’ll have to do a little research on this db) and one other vector database you choose based on research.  Examine the speed of indexing and querying as well as the memory usage. 
Tweaks to the System prompt. Use the one from the class example as a starting point. 
Try at least 2 different local LLMs.  Examples include Llama 2 7B and Mistral 7B.  You aren’t required to use these two specifically, however. 

Suggested Steps:

Collect and clean the data.  
If you’re using PDFs, review the output of whichever Python PDF library you’re using.  Is it what you expect? 
Do you want to pre-process the raw text in some way before indexing?  Perhaps remove extra white space or remove stop words?
Implement a driver Python script to execute your various versions of the indexing pipeline and to collect important data about the process (memory, time, etc).  Systematically vary the chunking strategies, embedding models, various prompt tweaks, choice of Vector DB, and choice of LLM. 
Develop a set of user questions that you give to each pipeline and qualitatively review the responses. 
Choose which pipeline you think works the best, and justify your choice. 
