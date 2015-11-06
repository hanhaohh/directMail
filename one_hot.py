from sklearn.preprocessing import OneHotEncoder
import itertools

# two example documents
docs = ["A A C", "B B D", "B C E"]

# split documents to tokens
tokens_docs = [doc.split(" ") for doc in docs]
print tokens_docs
# convert list of of token-lists to one flat list of tokens
# and then create a dictionary that maps word to id of word,
# like {A: 1, B: 2} here
all_tokens = itertools.chain.from_iterable(tokens_docs)
word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}
print word_to_id
# convert token lists to token-id lists, e.g. [[1, 2], [2, 2]] here
token_ids = [[word_to_id[token] for token in tokens_doc] for tokens_doc in tokens_docs]
print token_ids
# convert list of token-id lists to one-hot representation
vec = OneHotEncoder(n_values=len(word_to_id))
X = vec.fit_transform(token_ids)

print X.toarray()