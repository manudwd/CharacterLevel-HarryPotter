#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os, glob
books = glob.glob('*.txt')
comb = ""
for book in books:
    with open(book, encoding='utf8') as fp:
        comb+=fp.read()
        comb+='\n\n'

with open ('combined_books.txt', 'w', encoding='utf8') as fp:
    fp.write(comb)

