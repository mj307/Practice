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


'''
how the process will work:
- read data
- put into chroma/vector db
- when we aska question abt the data, only the relevant data will get passed thru
* doing this will reduce the amount of data the model will have to look thru in order to find the answer to our query
- we then pass this reduced dataset to our LLM
- we now ask the same exact question to the LLM that has the reduced dataset as the context
'''


# In[77]:


from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# In[78]:


def format_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs)


# In[79]:


template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
{context}
Question: {question}
Answer:"""


# In[81]:


custom_rag_prompt = PromptTemplate.from_template(template)


# In[83]:


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)
'''
the | character is used to "chain" together many functions
'''


# In[91]:


query = 'Who are the people that are not monitored by the party?'
for chunk in rag_chain.stream(query):
	print(chunk, end="", flush=True)


# In[92]:


query = 'Who make up the majority of Oceania\'s population?'
for chunk in rag_chain.stream(query):
	print(chunk, end="", flush=True)


# In[ ]:




