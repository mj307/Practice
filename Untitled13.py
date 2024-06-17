#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install chromadb')


# In[56]:


import transformers
import chromadb
from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma


# In[36]:


from langchain_community.document_loaders import PyPDFLoader


# In[65]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=100, add_start_index=True)
loader = PyPDFLoader("/Users/medhavijam/Desktop/book.pdf")
docs = loader.load()
all_splits = text_splitter.split_documents(docs)


# In[70]:


#all_splits[29:39]


# In[57]:


from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


# In[35]:


vectordb = Chroma.from_documents(documents=all_splits, embedding=embeddings, persist_directory="chroma_db")


# In[45]:


vectordb


# In[39]:


from langchain_community.llms import Ollama
llm = Ollama(model="llama3") 


# In[40]:


retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
# RetreivalQA retrieves data from the documents (whatever is passed into retriever) and is then passed into the llm


# In[72]:


query = "What does Winston drink all the time?"
docs = vectordb.similarity_search(query)
docs


# In[75]:


raw_results = vectordb.similarity_search_with_score(
    query,
    k=3
)
raw_results[0]


# things to fix: why is it giving m the same result multiple times
# the model is not able to search correctly
# only pass into the llm the top 5 or 10 or whatever occurances of a similarity found in the vectordb


# In[41]:


def test_rag(qa,query):
    print (f"Query: {query}")
    result = qa.run(query)
    print (f"Result: {result}")


# In[42]:


query = "Who is Big Brother?"
test_rag(qa,query)


# In[66]:


query = "What does Winston drink all the time?"
test_rag(qa,query)


# In[67]:


docs = vectordb.similarity_search(query)
print(f"Query: {query}")
print(f"Retrieved documents: {len(docs)}")
for doc in docs:
    doc_details = doc.to_json()['kwargs']
    print("Source: ", doc_details['metadata']['source'])
    print("Text: ", doc_details['page_content'], "\n")


# In[ ]:


# how the process will work:

