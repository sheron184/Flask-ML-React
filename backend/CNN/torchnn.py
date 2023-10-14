import torch
from torch import nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator
import spacy 
from transformers import AutoTokenizer
from torchvision.transforms import ToTensor
from transformers import TextClassificationPipeline
 
# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 2  # batch size for training

spacy_en = spacy.load('en_core_web_sm')

tokenize = lambda x: x.split() 
quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True) 
score = Field(sequential=False, use_vocab=False)

fields = {'quote':('q', quote), 'score':('s', score)}

train_data, test_data = TabularDataset.splits(
                                    path ='',
                                    train ='train.json',
                                    test=  'test.json',
                                    format ='json',
                                    fields = fields)


quote.build_vocab(train_data, 
                    max_size=10000,
                    min_freq=1)
# score.build_vocab(train_data, max_size=10, min_freq=1)

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=1,
    device="cpu"
)


class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        return self.fc(embedded)


model = TextClassificationModel(vocab_size=len(quote.vocab), embed_dim=100, num_class=1)
optimizer = optim.SGD(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()  # Assuming binary classification

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_iterator:
        optimizer.zero_grad()
        text = batch.q
        predictions = model(text)
        loss = criterion(predictions.squeeze(1), batch.s.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss / len(train_iterator):.4f}")


# Make predictions

# Preprocess the input text
input_text = 'Bel'
