"""
아이템 기반 KNN
""" 

import pandas as pd 
import numpy as np 

movies = pd.read_csv("./ml-latest-small/movies.csv")
ratings = pd.read_csv("./ml-latest-small/ratings.csv")

print(movies.shape)
print(ratings.shape)

# print(movies.head(3))
print(ratings.head(3))

# ratings.rating.describe()
# ratings.rating.value_counts()

ratings = ratings[['userId', 'movieId', 'rating']]
ratings_matrix = ratings.pivot_table('rating', index='userId', columns='movieId')

# 행렬 생성
rating_movies = pd.merge(ratings, movies, on='movieId')
ratings_matrix = rating_movies.pivot_table('rating', index='userId', columns='title').fillna(0)

#ratings_matrix.head(3)

# 유사도 산출
# cosine_similiry는 행을 기준으로 유사도 행렬을 산출함
# 지금은 행이 유저 base

ratings_matrix_T = ratings_matrix.transpose()
#ratings_matrix_T.head(3)

from sklearn.metrics.pairwise import cosine_similarity

item_sim = cosine_similarity(ratings_matrix_T, ratings_matrix_T)

item_sim_df = pd.DataFrame(data=item_sim, index=ratings_matrix.columns, columns=ratings_matrix.columns)

item_sim_df.shape
item_sim_df.head(5)

item_sim_df["Godfather, The (1972)"].sort_values(ascending=False)[:20]
item_sim_df["Inception (2010)"].sort_values(ascending=False)[:20]

# 유저의 성향 반영
def predict_rating(ratings_arr, item_sim_arr):
  ratings_pred = ratings_arr.dot(item_sim_arr) / np.array([np.abs(item_sim_arr).sum(axis=1)])
  return ratings_pred

ratings_pred = predict_rating(ratings_matrix, item_sim_df.values)
ratings_pred_matrix = pd.DataFrame(data=ratings_pred)
ratings_pred_matrix.columns = ratings_matrix.columns
ratings_pred_matrix.set_index(ratings_matrix.index, inplace=True)

ratings_pred_matrix.head(3)

from sklearn.metrics import mean_squared_error

## error => PASS
# def get_mse(pred, actual):
  # pred = pred[actual.nonzero()].flatten()
  # actual = actual[actual.nonzero()].flatten()

#   return mean_sqaured_error(pred, actual)
# get_mse(ratings_pred, ratings_matrix.values)

def predict_rating_topsim(ratings_arr, item_sim_arr, n=20):
  pred = np.zeros(ratings_arr.shape)

  for col in range(ratings_arr.shape[1]):
    top_n_items = [np.argsort(item_sim_arr[:, col])[:-n-1:-1]]
    for row in range(ratings_arr.shape[0]):
      pred[row, col] = item_sim_arr[col, :][top_n_items].dot(ratings_arr[row,:][top_n_items].T)
      pred[row, col] /= np.sum(np.abs(item_sim_arr[col, :][top_n_items]))

  return pred      
  
ratings_pred = predict_rating_topsim(ratings_matrix.values, item_sim_df.values, n=20)
ratings_pred

print(mean_squared_error(ratings_pred, ratings_matrix.values))

user_rating_id = ratings_matrix.loc[9,:]
user_rating_id[ user_rating_id > 0].sort_values(ascending=False)[:10]

def get_unseen_movies(ratings_matrix, userId):
  
  user_rating = ratings_matrix.loc[userId, :]
  already_seen = user_rating[ user_rating > 0].index.tolist()
  movie_list  = ratings_matrix.columns.tolist()
  unseen_list = [movie for movie in movie_list if movie not in already_seen]

  return unseen_list

def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
  recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]
  return recomm_movies

unseen_list = get_unseen_movies(ratings_matrix, 9)

recomm_movies = recomm_movie_by_userid(ratings_pred_matrix, 9, unseen_list, top_n=10)

## MF
def matrix_factorization(R, K, steps=200, learning_rate=0.01, r_lambda=0.01):

  num_users, num_items = R.shape
  np.random.seed(1)
  P = np.random.normal(scale=1./K, size=(num_users, K))
  Q = np.random.normal(scale=1./K, size=(num_items, K))

  prev_rmse = 10000
  break_count = 0

  non_zeros = [ (i, j, R[i,j]) for i in range(num_users) for j in range(num_items) if R[i,j] > 0]

  for step in range(steps):
    for i, j, r in non_zeros:
      eij = r - np.dot(P[i, :], Q[j, :].T)
      P[i, :] = P[i, :] + learning_rate*(eij*Q[j,:] - r_lambda*P[i, :])
      Q[j, :] = Q[j, :] + learning_rate*(eij*P[i,:] - r_lambda*Q[j, :])
    
    #rmse = mean_squared_error(R, P, Q, non_zeros)
    if (step % 10) == 0:
      print(step)
  
  return P, Q    

movies = pd.read_csv("./ml-latest-small/movies.csv")
ratings = pd.read_csv("./ml-latest-small/ratings.csv")
ratings = ratings[['userId', 'movieId', 'rating']]
ratings_matrix = ratings.pivot_table('rating', index='userId', columns='movieId')
rating_movies = pd.merge(ratings, movies, on='movieId')
ratings_matrix = rating_movies.pivot_table('rating', index='userId', columns='title').fillna(0)

P, Q = matrix_factorization(ratings_matrix.values, K=50, steps=100, learning_rate=0.01, r_lambda=0.01)  

pred_matrix = np.dot(P, Q.T)
ratings_pred_matrix = pd.DataFrame(data=pred_matrix, index=ratings_matrix.index, columns=ratings_matrix.columns)
#ratings_pred_matrix.head(3)

unseen_list = get_unseen_movies(ratings_matrix, 9)
recomm_movies = recomm_movie_by_userid(ratings_pred_matrix, 9, unseen_list, top_n=10)
recomm_movies 