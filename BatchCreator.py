# Create batch object for getting a tuple of tensors that will be used as
#the input to the model and the supervision data

class BatchCreator:
    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            x = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (x, y)


