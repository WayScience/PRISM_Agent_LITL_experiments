#!/usr/bin/env python
# coding: utf-8

# # PRISM 20Q2 and DepMap 24Q4 data retrieval
# 
# Retrieve **secondary-screen replicate-collapsed LFC**, **secondary-screen treatment information**, and **24Q4 model metadata** using `nbutils.prism_download`.
# 
# The helper resolves pinned Figshare article versions directly; no DepMap catalogue or browser step is needed. Existing files are reused only when their published MD5 matches. Set `OVERWRITE_EXISTING = True` to download fresh copies.
# 
# Each new file is downloaded to a temporary location, verified against Figshare's published MD5, and atomically moved into `data/`.

# In[1]:


from nbutils.pathing import repo_root
from nbutils.prism_download import download_prism_data


# In[2]:


data_path = repo_root() / "data"
OVERWRITE_EXISTING = False


# In[3]:


summary = download_prism_data(data_path, overwrite=OVERWRITE_EXISTING)
summary

