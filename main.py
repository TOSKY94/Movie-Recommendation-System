
import numpy as np
import pandas as pd
import difflib
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#print('loading movies db...')
try:
    #Data upload
    movies = pd.read_csv('IMDB_movies.csv')
    #print('loading complete...')

    #features selection
    #print('feature selection...')
    selected_features = ['Movie_Title','main_genre','side_genre','Actors','Director']
    for feature in selected_features:
        movies[feature]=movies[feature].fillna('')
        movies[feature]=movies[feature].str.replace(',',' ')

    #combine features
    combined_features = movies['Movie_Title']+' '+movies['main_genre']+' '+movies['side_genre']+' '+movies['Actors']+' '+movies['Director']

    #stemming
    #print('stemming...')
    stemmer = SnowballStemmer('english')
    def stemming_tokenizer(str_input):
        words = re.sub(r'[^a-zA-Z]{2,}', ' ', str_input).lower().split()
        words = [stemmer.stem(word) for word in words]
        return ' '.join(words)

    stemmed_features = combined_features.apply(stemming_tokenizer)

    #vectorizing
    #print('vectorizing...')
    vectorizer = TfidfVectorizer(stop_words='english')
    feature_vectors = vectorizer.fit_transform(stemmed_features)

    #similarity
    #print('creating similarity scores...')
    similarity = cosine_similarity(feature_vectors)

    #get recommendation
    def recommendation(movie_name):
        movie_titles =movies['Movie_Title'].tolist()
        movie_matches = difflib.get_close_matches(movie_name.title(), movie_titles)
        movie_idx = movies[movies['Movie_Title']==movie_matches[0]].index.values[0]
        similarity_scores = list(enumerate(similarity[movie_idx]))
        sorted_movies = sorted(similarity_scores, key=lambda x:x[1], reverse = True)
        
        recommendation = []
        for idx, score in sorted_movies[:20]:
            recommendation.append(movies[movies.index==idx]['Movie_Title'].values[0])

        return recommendation
    
    movie_name = input('Enter movie name: ')
    recommended_movies = recommendation(movie_name)
    print('\nRecommended movies')
    print('-------------------------')
    for idx, movie in enumerate(recommended_movies):
        print(f'{movie}')
except Exception as e:
    print('error occured',e)