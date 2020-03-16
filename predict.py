import torch
import numpy as np
import torch.nn.functional as F
from Classifier import Classifier
import load_data
import utility

TEXT, vocab_size, train_iter, valid_iter = load_data.load_data()

model = utility.instantiate_model(vocab_size)

#predict a post for testing purpose
def predict(text):
    model.load_state_dict(torch.load('model.pt'))
    model.eval()

    device = utility.get_device()

    text = TEXT.preprocess(utility.clean_post(text))

    if (len(text) != 0): 

        text = [[TEXT.vocab.stoi[x] for x in text]]
        text = np.asarray(text)
        text_tensor = torch.LongTensor(text).to(device)

        output = model(text_tensor)
        output = F.softmax(output,1)

        #probabilities between classes  
        prob = [utility.truncate_decimal(torch.max(output[:,i]).item())
            for i in range (2)]

        if (prob[0] < prob[1]):
            print("Positive post")
        elif (prob[0] > prob[1]):
            print("Negative post")
        else:
            print("Unclassified")
    else: 
        print("Unclassified")

while True:
    text = input("Type your post: (Type Q to quit)\n")
    if (text !='Q'):
        predict(text)
    if (text == 'Q'):
        break;    

   


