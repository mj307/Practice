#!/usr/bin/env python
# coding: utf-8

# In[2]:


import chromadb
from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader


# In[23]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=100, add_start_index=True)
loader = PyPDFLoader("/Users/medhavijam/Desktop/adobe.pdf")
docs = loader.load()
all_splits = text_splitter.split_documents(docs)


# In[25]:


all_splits[5:10]


# In[20]:


#!pip install -U langchain-huggingface


# In[26]:


from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


# In[27]:


vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chromadb")


# In[28]:


from langchain_community.llms import Ollama
llm = Ollama(model="llama3") 


# In[29]:


from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# In[30]:


def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)


# In[31]:


template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
{context}
Question: {question}
Answer:"""


# In[32]:


custom_rag_prompt = PromptTemplate.from_template(template)
retriever = vectordb.as_retriever()


# In[33]:


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)


# In[34]:


query = 'Who are the speakers?'
for chunk in rag_chain.stream(query):
	print(chunk, end="", flush=True)


# In[35]:


query = 'What is the main topic of the conversation?'
for chunk in rag_chain.stream(query):
	print(chunk, end="", flush=True)


# In[36]:


query = 'What was Adobe\'s revenue?'
for chunk in rag_chain.stream(query):
	print(chunk, end="", flush=True)


# In[37]:


query = 'What are Adobe\'s flagship products?'
for chunk in rag_chain.stream(query):
	print(chunk, end="", flush=True)


# In[38]:


query = 'What are Adobe\'s main products?'
for chunk in rag_chain.stream(query):
	print(chunk, end="", flush=True)


# In[39]:


query = 'When did they celebrate the five-year anniversary of Adobe Experience Platform?'
for chunk in rag_chain.stream(query):
	print(chunk, end="", flush=True)


# In[40]:


query = 'What is the Acrobat AI Assistant? '
for chunk in rag_chain.stream(query):
	print(chunk, end="", flush=True)


# In[43]:


query = 'What will the AI Assistant be add-ons to? Give me their specific names.'
for chunk in rag_chain.stream(query):
	print(chunk, end="", flush=True)


# In[ ]:




