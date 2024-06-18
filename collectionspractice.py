#!/usr/bin/env python
# coding: utf-8

# In[3]:


import chromadb


# In[21]:


client = chromadb.PersistentClient(path="chromapractice")


# In[22]:


client.heartbeat()


# In[23]:


from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


# In[24]:


collection = client.create_collection(name="my_collection1")
collection = client.get_collection(name="my_collection1")


# In[35]:


collection = client.get_or_create_collection(name="test")


# In[36]:


collection.peek()


# In[28]:


collection.count()


# In[37]:


collection = client.create_collection(
        name="collection_name",
        metadata={"hnsw:space": "cosine"} # l2 is the default
    )


# In[39]:


collection.peek()


# In[41]:


collection.add(
    documents=["lorem ipsum...", "doc2", "doc3"],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}],
    ids=["id1", "id2", "id3"]
)


# In[42]:


collection.peek()


# In[ ]:




