'''
https://towardsdatascience.com/how-to-build-a-movie-recommender-system-in-python-using-lightfm-8fa49d7cbe3b

LightFM is a Python implementation of a number of popular recommendation algorithms. 
LightFM includes implementations of BPR and WARP ranking losses(A loss function is a measure 
of how good a prediction model does in terms of being able to predict the expected outcome.).

BPR: Bayesian Personalised Ranking pairwise loss: It maximizes the prediction difference between 
	a positive example and a randomly chosen negative example. It is useful when only positive interactions are present.

WARP: Weighted Approximate-Rank Pairwise loss: Maximises the rank of positive examples 
	by repeatedly sampling negative examples until rank violating one is found
'''


import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

# data prep
data = fetch_movielens(min_rating = 4.0)

print(repr(data['train']))
print(repr(data['test']))

model = LightFM(loss = 'warp')
model.fit(data['train'], epochs=30, num_threads=2)

# reo engine
def simple_recommendation(model, data, user_ids):

	n_users, n_items = data['train'].shape

	for user_id in user_ids:
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
		scores = model.predict(user_id, np.arange(n_items))
		top_items = data['item_labels'][np.argsort(-scores)]

		print("User %s" % user_id)		
		print("	Known positives:")

		for x in known_positives[:10]:
			print("		 %s" % x)

		print("	Recommended:")

		for x in top_items[:10]:
			print("		 %s" % x)

# 
simple_recommendation(model, data, [3, 4, 25, 451])	