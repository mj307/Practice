#!/usr/bin/env python
# coding: utf-8

# In[4]:


import chromadb
from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


# In[2]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=100, add_start_index=True)
loader = PyPDFLoader("/Users/medhavijam/Desktop/brain.pdf")
docs = loader.load()
all_splits = text_splitter.split_documents(docs)


# In[3]:


all_splits[10:17]


# In[5]:


vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="summarydb")


# In[6]:


from langchain_community.llms import Ollama
llm = Ollama(model="llama3") 


# In[7]:


from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# In[8]:


def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)


# In[9]:


template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer as concise as possible.
{context}
Question: {question}
Answer:"""


# In[10]:


custom_rag_prompt = PromptTemplate.from_template(template)
retriever = vectordb.as_retriever()


# In[11]:


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)


# In[12]:


query = 'Summarize the document'
for chunk in rag_chain.stream(query):
	print(chunk, end="", flush=True)


# In[14]:


query = 'What are the different topics covered in the document? Be specific.'
for chunk in rag_chain.stream(query):
	print(chunk, end="", flush=True)


# In[15]:


query = 'Give me 20 study questions about the content.'
for chunk in rag_chain.stream(query):
	print(chunk, end="", flush=True)


# In[ ]:




