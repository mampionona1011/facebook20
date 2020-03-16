import torch.nn as nn

#Define the post classification model

class Classifier(nn.Module):
        def __init__(self, output_size, hidden_size, vocab_size, embedding_length):
                super(Classifier, self).__init__()
                #parameters
                self.output_size = output_size
                self.hidden_size = hidden_size
                self.vocab_size = vocab_size
                self.embedding_length = embedding_length

                #layers of the architecture
                self.embedding = nn.Embedding(vocab_size, embedding_length)
                self.encoder = nn.LSTM(embedding_length, hidden_size,num_layers=1)
                self.predictor = nn.Linear(hidden_size, output_size)

        def forward(self, inputs):
                embedded = self.embedding(inputs)
                outputs,_ = self.encoder(embedded)

                return self.predictor(outputs[-1,:,:])
