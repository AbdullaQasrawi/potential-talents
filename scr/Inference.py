#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import math

pd.set_option('display.max_columns', 100)
pd.set_option('display.expand_frame_repr', True)
pd.options.display.max_colwidth = 100000


class JobSearch:

    def __init__(self, file_path= "..\Data", file_name= "potential-talents.csv"):
        self.talents = pd.read_csv(f"{file_path}/{file_name}")
        self.talents2 = self.talents.copy()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.employee_number = ''



    def clean_text(self, job_titles):
        employee_titles = []
        for i in range(0, len(job_titles)):
            titles = re.sub('[^a-zA-Z]', ' ', job_titles[i])
            titles = titles.lower()
            titles = titles.split()
            titles = ' '.join(titles)
            employee_titles.append(titles)
        return employee_titles

    def get_job_location(self):
        job_location = input('best location: ')
        if job_location:
            return [job_location]
        else:
            return []

    def get_job_titles(self):
        job_titles = input('write the jobe describtion or skils you are loking for: ')
        if job_titles:
            return [job_titles]
        else:
            return []
        
    def set_employee_number(self):
        self.employee_number = input('write the the s: ')
        
    def embedding(self, embedded_vector):
        return self.model.encode(embedded_vector)
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    

    
    def get_similarity(self, data, titles_embeddings, key_title_embeddings, key_title, location_embeddings, key_location_embeddings, key_location):
        

        
        title_similairt = cosine_similarity(titles_embeddings, key_title_embeddings) if key_title else None
        location_similairt = cosine_similarity(location_embeddings, key_location_embeddings) if key_location else None
        employee_number = self.employee_number
        
        if not employee_number:
            if not key_location:
                if data['similarity'].sum() == 0:
                    data['similarity'] = pd.Series(title_similairt.ravel())
                else:
                    data['similarity'] = (data['similarity'] + pd.Series(title_similairt.ravel()))/2
            elif not key_title:
                if data['similarity'].sum() == 0:
                    data['similarity'] = pd.Series(location_similairt.ravel())
                else:
                    data['similarity'] =  (data['similarity'] + pd.Series(location_similairt.ravel()))/2
            else:  
                combin_sim = (location_similairt + title_similairt)/2
                if data['similarity'].sum() == 0:
                    data['similarity'] = combin_sim
                else:
                    data['similarity'] = (data['similarity'] + pd.Series(combin_sim.ravel()))/2

            ## lets combine the similarity by averaging the values
            data['fit'] = [self.sigmoid(sim) for sim in data['similarity']]
            return data.sort_values(by=['fit', 'connection'], ascending=False).head(15)

        else:
            data.loc[data['id'] == int(employee_number), 'selected'] = 1
            ## lets extract the index 
            idx= data[data['id'] == int(employee_number)].index[0]
            selecte_title = titles_embeddings[idx]
            selected_location = location_embeddings[idx]

            select_emp_title_sim = cosine_similarity(titles_embeddings, selecte_title.reshape(1,-1))
            select_emp_location_sim = cosine_similarity(location_embeddings, selected_location.reshape(1,-1))

            ### lets average with previous state
            combin_emp_sim = (select_emp_title_sim + select_emp_location_sim)/2
            data['similarity'] = (0.4*data['similarity'] +  0.6*pd.Series(combin_emp_sim.ravel()))/2
            ## lets combine the similarity by averaging the values
            data['fit'] = [self.sigmoid(sim) for sim in data['similarity']]

            employee_number = ''
            return data.sort_values(by=['fit', 'connection'], ascending=False).head(15)


user_file_path = input("Enter the file path: ")
user_file_name = input("Enter the file name: ")
if not user_file_path:
    search = JobSearch()
else:
    search = JobSearch(user_file_path, user_file_name)
    

key_location = search.get_job_location()
key_location_embedding = search.embedding(key_location)

key_title = search.get_job_titles()
key_title_embedding = search.embedding(key_title)

job_title = search.clean_text(search.talents["job_title"])
job_title_embedding = search.embedding(job_title)

job_location = search.clean_text(search.talents["location"])
job_location_embedding = search.embedding(job_location)

print(search.get_similarity(data=search.talents, titles_embeddings=job_title_embedding, key_title_embeddings=key_title_embedding, key_title=key_title, 
               location_embeddings=job_location_embedding, key_location_embeddings=key_location_embedding, key_location=key_location))
               

search.set_employee_number()

print(search.get_similarity(data=search.talents, titles_embeddings=job_title_embedding, key_title_embeddings=key_title_embedding, key_title=key_title, 
               location_embeddings=job_location_embedding, key_location_embeddings=key_location_embedding, key_location=key_location))
               