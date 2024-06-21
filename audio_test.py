#!/usr/bin/env python
# coding: utf-8

# In[1]:


audio_call = "/Users/medhavijam/Desktop/call_108.mp3"


# In[ ]:


'''
goals:
- clean up audio file, remove background noises, make it more clear
- 
'''


# In[10]:


#!pip install torch torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html


# In[9]:


#!pip install deepfilternet


# In[6]:


from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file


# In[8]:


get_ipython().system('pip install librosa')


# In[11]:


import librosa
import noisereduce as nr
import numpy as np

def clean_audio(input_file, output_file):
    # Load audio file
    y, sr = librosa.load(input_file)

    # Reduce noise
    noisy_part = y[10000:80000]  # Example: Analyze a portion of the audio for noise profile
    noise_profile = np.mean(np.square(noisy_part))
    y = nr.reduce_noise(y, y, verbose=False)

    # Save the cleaned audio
    librosa.output.write_wav(output_file, y, sr=sr)

# Example usage:
input_file = audio_call  # Replace with your input audio file
output_file = '/Users/medhavijam/Desktop'  # Replace with desired output path
clean_audio(input_file, output_file)


# In[ ]:




