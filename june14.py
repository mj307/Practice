#!/usr/bin/env python
# coding: utf-8

# In[1]:


import langchain
from langchain_community.llms import Ollama

llm = Ollama(
    model="llama3"
) 
llm.invoke("Teach me how to say hello in Hindi")


# In[3]:


get_ipython().system('pip install llama_hub')


# In[5]:


get_ipython().system('pip install llama-index-readers-web')


# In[9]:


get_ipython().system('pip install llama-index-readers-wikipedia')


# In[14]:


from llama_index.readers.wikipedia import WikipediaReader
loader = WikipediaReader()
documents = loader.load_data(pages=['Berlin', 'Rome', 'Tokyo', 'Canberra', 'Santiago'])


# In[20]:


from llama_index.core import VectorStoreIndex
#documents
index = VectorStoreIndex.from_documents(documents)
index
#query_engine = index.as_query_engine()
#response = query_engine.query("How many people live in Berlin")


# In[ ]:




