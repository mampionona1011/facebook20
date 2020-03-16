import utility
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator
from BatchCreator import BatchCreator
from torchtext import vocab

def load_data():
    tokenizer = lambda x: x.split()

    #define fields
    TEXT = Field(sequential = True, tokenize = tokenizer, lower = True)
    LABEL = Field(sequential = False, use_vocab = False)
    
    fields = [("Text", TEXT),("Category", LABEL)]

    #create dataset
    data = TabularDataset(path = 'dataset/train.csv',format = 'csv',fields = fields,skip_header = True)

    #train validation split
    trn, vld = data.split(split_ratio = 0.75)

    #build vocabulary 
    TEXT.build_vocab(trn,vld)

    device = utility.get_device()

    #load data in batch
    train_iter, valid_iter = BucketIterator.splits((trn,vld)
                                            ,batch_sizes = (64,64)
                                            ,device = device
                                            ,sort_key = lambda x: len(x.Text)
                                            ,sort_within_batch = True
                                            ,repeat = False)
    vocab_size = len(TEXT.vocab)

    #wrap iterator
    train_dl = BatchCreator(train_iter, "Text", "Category")
    valid_dl = BatchCreator(valid_iter, "Text", "Category")

    return TEXT, vocab_size, train_dl, valid_dl
