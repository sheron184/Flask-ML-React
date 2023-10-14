tokenize = lambda x: x.split() 

quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True) 
score = Field(sequential=False)

fields = {'quote':('q', quote), 'score':('s', score)}

train_data, test_data= TabularDataset.splits(
                                    path ='',
                                    train ='train.json',
                                    test=  'test.json',
                                    format ='json',
                                    fields = fields
                                )


spacy_en = spacy.load('en_core_web_sm')
def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


quote.build_vocab(train_data, 
                    max_size=10000,
                    min_freq=1)
score.build_vocab(train_data, max_size=10000, min_freq=1)

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=2,
    device="cpu"
)