import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings('ignore')
from ast import literal_eval
pd.set_option('max_colwidth', 100)

movies = pd.read_csv('tmdb_5000_movies.csv')
#print(movies.shape)

cols = ['id', 'title', 'genres', 'vote_average', 'vote_count',
        'popularity', 'keywords', 'overview']

movie_df = movies[cols]
movie_df['genres'] = movie_df['genres'].apply(literal_eval)
movie_df['keywords'] = movie_df['keywords'].apply(literal_eval)

movie_df['genres'] = movie_df['genres'].apply(lambda x: [ y['name'] for y in x])
movie_df['keywords'] = movie_df['keywords'].apply(lambda x: [ y['name'] for y in x])

#print(movie_df[['genres', 'keywords']][:1])

from sklearn.feature_extraction.text import CountVectorizer

movie_df['genre_literal'] = movie_df['genres'].apply(lambda x: (' ').join(x))

#print(movie_df.genre_literal.head(5))

count_vect = CountVectorizer(min_df=0, ngram_range=(1,2))
genre_mat = count_vect.fit_transform(movie_df['genre_literal']) 

#print(genre_mat.shape)

from sklearn.metrics.pairwise import cosine_similarity

genre_sim = cosine_similarity(genre_mat, genre_mat)

print(genre_sim.shape)
print(genre_sim[:2])

genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1]
print(genre_sim_sorted_ind[:1])

# 평점 반영 없이 유사도만
def find_sim_movie(df, sorted_ind, title_name, top_n=10):

    title_movie = df[df['title'] == title_name]
    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, :(top_n)]

#    print(similar_indexes)
    similar_indexes = similar_indexes.reshape(-1)

    return df.iloc[similar_indexes]

similar_movies = find_sim_movie(movie_df, genre_sim_sorted_ind, 'The Godfather', 10)
similar_movies[['title', 'vote_average']]

## 평점 반영
percentile = 0.6
m = movie_df['vote_count'].quantile(percentile)
C = movie_df['vote_average'].mean()

def weighted_vote_average(record):
    v = record['vote_count']
    R = record['vote_average']
    #가중평점
    return ( (v/(v+m)) * R ) + ( (m/(m+v)) * C )   
        
movie_df['weighted_vote'] = movie_df.apply(weighted_vote_average, axis=1)

movie_df.head(1)

# 평점 반영 및 유사도
def find_sim_movie_with_score(df, sorted_ind, title_name, top_n=10):

    title_movie = df[df['title'] == title_name]
    title_index = title_movie.index.values

    similar_indexes = sorted_ind[title_index, :(top_n*2)]
    similar_indexes = similar_indexes.reshape(-1)
   # similar_indexes = similar_indexes[similar_indexes != title_index]

    return df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]

similar_movies = find_sim_movie_with_score(movie_df, genre_sim_sorted_ind, 'The Godfather', 10)
similar_movies[['title', 'vote_average', 'weighted_vote']]
