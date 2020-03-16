import re
import torch
from Classifier import Classifier

#parameters
output_size = 2
hidden_size = 32
embedding_length = 300

#instantiate model
def instantiate_model(vocab_size):
    return Classifier(output_size, hidden_size, vocab_size, embedding_length)


#define available device
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#extract stopwords
def stop_words():
    with open('stopwords.txt') as file:
        return [line.rstrip('\n') for line in file]

#pre-processing text
def clean_post(text):
    post = []

    #remove link, mail, HTML tags, non alphanumeric characters
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ',text)
    text = re.sub(r'[\w.-]+@[\w.-]+',' ', text)
    text = re.sub(r'<[^<]+?>', ' ', text) 
    text = re.sub(r'[^A-Za-z\']+', ' ', text) 
        
    for word in text.split():
        if word.lower() not in stop_words():
            post.append(word.lower())
    return post

#truncate decimal
def truncate_decimal(number):
    return re.sub(r'^(\d+\.\d{,1})\d*$',r'\1',str(number))

