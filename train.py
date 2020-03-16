import torch
import torch.nn as nn
import torch.optim as optim
import load_data
from utility import instantiate_model 

print('Loading data...')
TEXT, vocab_size, train_iter, valid_iter = load_data.load_data()

print('Done!')

model = instantiate_model(vocab_size)

print(model)

#define optimizer and loss
learning_rate = 0.0001
optimizer = optim.Adam(model.parameters(), learning_rate)
criterion = nn.CrossEntropyLoss()

#train the model
print('Training the model...')

epochs = 50 #this value can be modified until to get the best model
for epoch in range(epochs):
    training_loss = 0.0
    valid_loss = 0.0

    #training mode   
    model.train() 
    for x, y in train_iter:
        optimizer.zero_grad()

        preds = model(x)
        loss = criterion(preds,y)

        training_loss += loss.item()     

        loss.backward()
        optimizer.step()

        trn_loss = training_loss / len(train_iter)
       
    #evaluation mode
    model.eval()
    for x, y in valid_iter:
        preds = model(x)
        loss = criterion(preds,y)

        valid_loss += loss.item()

        vld_loss = valid_loss / len(valid_iter)

    #save model
    best_vld_loss = float('inf')
    if vld_loss < best_vld_loss:
        best_vld_loss = vld_loss
        torch.save(model.state_dict(),'model.pt')
        
    print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}'
          .format(epoch + 1, trn_loss, vld_loss))

print('Training successful!')
