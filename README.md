# potential-talents
This project aims to find the probability of suitability of an employees for a specific job based on skills, location and number of connections on linkedin.  

# Description  
This Python script provides a class JobSearch to search for potential candidates from a CSV file of job seekers. 
It allows to search based on the job location and job title or skills. also it take the effect of selecting an employee into conserderation for new recomendation requist.  
the code mainly used Pre trained Sentence transforemer (all-MiniLM-L6-v2) which is a high-dimensional vector representation of a sentence, and its give us good results. 
The JobSearch class uses natural language processing (NLP) techniques to clean the job titles and search for relevant candidates. 
It also uses cosine similarity to compute the similarity between the job description, locations and the skills and locations
we allredy have from talents profiles.  

the model accumulating similarities continuously and more weights given for new record comparing with preselected candidates, and this weights selected after many trials and errors.

# Model Structure  
the model contains many functions that work together in order to return the final required results. below a brief description for each function.  

### clean_text:  
remove any extra spaces, convert to lower case and removing punctuations.

### get_job_location, get_job_titles and set_employee_number  
to give the use the ability to write its preferences and take this data into consideration in the final result.  

### embedding:  
convert the text into high-dimensional vector using Pre trained Sentence transforemer (all-MiniLM-L6-v2).  

### sigmoid:  
convert the similarity to probability.  

### get_similarity:  
its the main function that compute the similarity between vectors using cosine similarity and the convert it to a probability that filled in fit variable the its return top 15 preffered talents.

# Installation  
Clone or download the repository.  
Install the required Python packages by running pip install -r requirements.txt.

# Usage  
Create an instance of the JobSearch class.  
Call the get_job_location() method to write the prefered job location. You can leave it blank if you don't have any preference.  
Call the get_job_titles() method to write the prefered job title or skills you are looking for.  
Call the set_employee_number() method to set the employee number for an employee that selected to be employed.  
Call the get_similarity() method to get the potential candidates.  

# Note  
You can specify path director and file name for your data or just use the prededined paths without changing any thing just keep the structure of folders as it is.
