#!/usr/bin/env python
# coding: utf-8

# In[2]:


import langchain
from langchain import HuggingFaceHub


# In[34]:


#llm = HuggingFaceHub(repo_id='google/flan-t5-xl', huggingfacehub_api_token='google/flan-t5-xl')
prompt = "Alice has a parrot. What animal is Alice's pet?"
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
huggingfacehub_api_token = 'api key'
llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", huggingfacehub_api_token=huggingfacehub_api_token)
# repo_id is the model type
#google/flan-t5-xxl
output = llm.invoke(prompt)
print(output)


# In[32]:


from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# The embeddings model takes a text as an input and outputs a list of floats
text = "Alice has a parrot. What animal is Alice's pet?"
text_embedding = embeddings.embed_query(text)


# In[21]:


#text_embedding


# In[26]:


from langchain import PromptTemplate
template = "What is a good name for a company that makes {product}?"
prompt_template = PromptTemplate(
    input_variables = ['product'],
    template=template,
)
prompt_template.format(product='colorful socks')


# In[27]:


from langchain.chains import LLMChain
chain = LLMChain(llm=llm,
                 prompt=prompt_template)
# note : for some reason chain isn't allowing prompt=prompt. 
# i had to change prompt to prompt_template in the code above for it to work
chain.run("colorful socks")


# In[30]:


second_prompt = PromptTemplate(
    input_variables = ['company_name'],
    template = "Write a catchphrase for the following company : {company_name}",
)

chain_two = LLMChain(llm=llm, prompt = second_prompt)

from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[chain,chain_two], verbose=True)

catchphrase = overall_chain.run("colorful socks")
catchphrase


# In[37]:


from langchain_community.llms import Ollama

llm = Ollama(
    model="llama3"
) 
llm.invoke("Tell me a joke")


# In[38]:


template = "What is a good name for a company that makes {product}?"
prompt_template = PromptTemplate(
    input_variables = ['product'],
    template=template,
)
# prompt_template.format(product='colorful socks')
chain = LLMChain(llm=llm,
                 prompt=prompt_template)
second_prompt = PromptTemplate(
    input_variables = ['company_name'],
    template = "Write a catchphrase for the following company : {company_name}",
)

chain_two = LLMChain(llm=llm, prompt = second_prompt)

from langchain.chains import SimpleSequentialChain
overall_chain = SimpleSequentialChain(chains=[chain,chain_two], verbose=True)

catchphrase = overall_chain.run("colorful socks")
catchphrase


# In[ ]:




