from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

## 반드시 user_id, item_id, rating 순서로, 3개만 있어야 함
data = Dataset.load_builtin('ml-100k')
trainset, testset = train_test_split(data, test_size=.25, random_state=0)

algo = SVD()
algo.fit(trainset)

pred = algo.test(testset)
print(type(pred), len(pred))
pred[:10]

# 데이터 개별 접근
[ (pred.uid, pred.iid, pred.est) for pred in pred[:3]]

# 예측시 문자열로 입력
uid = str(120)
iid = str(282)
pred_res = algo.predict(uid, iid)
print(pred_res)
print(accuracy.rmse(pred))

import pandas as pd 

ratings = pd.read_csv("./ml-latest-small/ratings.csv")
ratings.to_csv("./ml-latest-small/ratings_noh.csv", index=False, header=False)

from surprise import Reader

#### error
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))
data = Dataset.load_from_file('./ml-latest-small/ratings_noh.csv', reader=reader)
trainset, testset = train_test_split(data, test_size=.25, random_state=0)
algo = SVD(n_factor=50, random_stat=0)
algo.fit(trainset)
pred = algo.test(testset)
print(accuracy.rmse(pred))

import pandas as pd 
from surprise import Reader, Dataset 

ratings = pd.read_csv("./ml-latest-small/ratings.csv")
reader = Reader(rating_scale=(0.5, 5.0))

# 순서 중요
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=.25, random_state=0)

algo = SVD(n_factors=50, random_state=0)
algo.fit(trainset)
predictions = algo.test(testset)
accuracy.rmse(predictions)

# cv & gridsearch
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

ratings = pd.read_csv("./ml-latest-small/ratings.csv")
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# cv
algo = SVD(n_factors=50, random_state=0)
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# gridsearch
param_grid = {'n_epochs': [20,40,60], 'n_factors': [50, 100, 200]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

# for unseen data
from surprise.dataset import DatasetAutoFolds

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
algo = SVD(n_factors=50, random_state=0)
algo.fit(data) # error if train_split is not processed

reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))
data_folds = DatasetAutoFolds(ratings_file='./ml-latest-small/ratings_noh.csv', reader=reader)
trainset = data_folds.build_full_trainset()

algo = SVD(n_epochs=20, n_factors=50, random_state=0)
algo.fit(trainset)

movies = pd.read_csv('./ml-latest-small/movies.csv')
movieIds = ratings[ratings['userId']==9]['movieId']

# 42가 있는지 확인
if movieIds[movieIds == 42].count() == 0:
  print('No rating for movie42')

print(movies[movies['movieId'] == 42])

uid = str(9)
iid = str(42)
pred = algo.predict(uid, iid, verbose=True)
print(pred)

def get_unseen_surprise(ratings, movies, userId):
    
    seen_movies = ratings[ratings['userId'] == userId]['movieId'].tolist() 
    total_movies = movies['movieId'].tolist()
    unseen_movies = [movie for movie in total_movies if movie not in seen_movies]
    print(len(seen_movies), len(unseen_movies), len(total_movies))

    return unseen_movies

unseen_movies = get_unseen_surprise(ratings, movies, 9) # a list
 
def recomm_movies_by_surprise(algo, userId, unseen_movies, top_n=10): 
    predictions = [algo.predict(str(userId), str(movieId)) for movieId in unseen_movies]

    def sortkey_est(pred):
        return pred.est 

    predictions.sort(key=sortkey_est, reverse=True)    
    top_predictions = predictions[:top_n]

    top_movie_ids = [ int(pred.iid) for pred in top_predictions]
    top_movie_rating = [ pred.est for pred in top_predictions]
    top_movie_titles = movies[movies.movieId.isin(top_movie_ids)]['title']

    top_movie_pred = [ (id, title, rating) for id, title, rating in zip(top_movie_ids, top_movie_titles, top_movie_rating )]

    return top_movie_pred

    
unseen_movies = get_unseen_surprise(ratings, movies, 9)
top_movie_pred = recomm_movies_by_surprise(algo, 9, unseen_movies, top_n=10)

top_movie_pred

for top_movie in top_movie_pred:
    print(top_movie[1], ":", top_movie[2])