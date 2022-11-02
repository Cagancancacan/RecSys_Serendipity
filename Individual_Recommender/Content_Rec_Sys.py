from IPython.display import display
import pandas as pd
import numpy as np


def select_user(expert_users, ratings_genres_df):
    random_user = expert_users.sample()
    random_user = random_user['user']

    seen_movies = pd.merge(random_user, ratings_genres_df, how='left', left_on='user', right_on='user')

    return seen_movies


def select_user_id(expert_users, ratings_genres_df):
    random_user = expert_users.sample()

    random_user = random_user['user']
    user = random_user.copy()
    user.index = [0]
    user = user[0]

    seen_movies = pd.merge(random_user, ratings_genres_df, how='left', left_on='user', right_on='user')

    return seen_movies, user


def get_favourite_movies(seen_movies):
    return seen_movies.loc[seen_movies['rating'] == np.max(seen_movies['rating'])]


def get_genre_set(ratings_genres_df):
    genre_set = set()
    for genres in ratings_genres_df.genres:
        genre_set.update(genres.split('|'))

    return genre_set


def get_powerset(items):
    powerset = list(items)
    n = len(powerset)
    return [[powerset[k] for k in range(n) if i & 1 << k] for i in range(2 ** n)]


def get_favourite_genres_powerset(genre_set, favourite_movies, num_genres=3):
    genre_set_list = list(genre_set)
    genre_count = np.zeros(len(genre_set_list))

    for genre in favourite_movies.genres:
        genre_list = genre.split('|')
        for i in range(len(genre_list)):
            for j in range(len(genre_set_list)):
                if genre_list[i] == genre_set_list[j]:
                    genre_count[j] += 1

    favourite_genres = []
    while len(favourite_genres) < num_genres:
        if np.max(genre_count) == 0:
            break

        fav_genre = genre_set_list[np.argmax(genre_count)]
        genre_count[np.argmax(genre_count)] = 0

        if fav_genre == "(no genres listed)":
            continue

        favourite_genres.append(fav_genre)

    powerset = get_powerset(favourite_genres)
    display(powerset)
    return powerset


def get_movies_with_genres(find_genres, genre_df):
    items = []

    for genres in find_genres:
        if not genres:
            continue
        for genre in genres:
            temp = genre_df.loc[(genre_df[genre] == True)]
            items.extend(temp.item)

    out = genre_df[genre_df['item'].isin(items)]
    out.index = np.arange(len(out.index))

    return out


def get_unseen_movies_ratings_genres(movies_df, ratings_df, movies_with_genres, seen_movies):
    unseen_movies = list(pd.concat([movies_df.item, seen_movies.item]).drop_duplicates(keep=False))
    unseen_movies_ratings = ratings_df[ratings_df['item'].isin(unseen_movies)]
    unseen_movies_with_genres = movies_with_genres[movies_with_genres['item'].isin(unseen_movies)]

    return unseen_movies, unseen_movies_ratings, unseen_movies_with_genres


def get_average_ratings(unseen_movies_with_genres, average_ratings_df):
    return average_ratings_df.loc[average_ratings_df['item'].isin(unseen_movies_with_genres.item)]


def get_top_movies(average_ratings_df):
    top_rating = np.max(average_ratings_df.average_rating)

    top_movies = []
    for item in average_ratings_df.index:
        if average_ratings_df.loc[item].average_rating < top_rating - (top_rating / 10):
            continue
        top_movies.append(item)

    return top_movies


def get_favourite_tags(favourite_movies, tags_df):
    favourite_movies_list = list(favourite_movies.item)

    favourite_tags = set()
    for item in favourite_movies_list:
        movie_tags = tags_df.loc[tags_df['item'] == item]
        favourite_tags.update(list(movie_tags.tag))

    return favourite_tags


def dice_coefficient(fav_tag, movie_tag):
    return (2 * len(fav_tag.intersection(movie_tag))) / (len(fav_tag) + len(movie_tag))


def find_k_best(k, movie_similarity, movies_df):
    movie_list = []
    for i in range(k + 1):
        recommend_movie = max(movie_similarity, key=movie_similarity.get)
        movie_list.append(recommend_movie)
        movie_similarity[recommend_movie] = 0

    return movies_df[movies_df['item'].isin(movie_list)]


def get_movie_recommendations(top_movies, tags_df, favourite_tags, movies_df, k):
    movie_similarity = {}
    for item in top_movies:
        tag_set = set()
        movie_tags = tags_df.loc[tags_df['item'] == item]
        tag_set.update(list(movie_tags.tag))
        movie_similarity[item] = dice_coefficient(favourite_tags, tag_set)

    return find_k_best(k, movie_similarity, movies_df)


def content_recommender(expert_users_df, ratings_genres_df, movies_df, ratings_df, genre_df, tags_df,
                        average_ratings_df):
    k = 10

    seen_movies = select_user(expert_users_df, ratings_genres_df)

    favourite_movies = get_favourite_movies(seen_movies)

    genre_set = get_genre_set(ratings_genres_df)

    genre_list = get_favourite_genres_powerset(genre_set, favourite_movies)

    movies_with_genres = get_movies_with_genres(genre_list, genre_df)

    unseen_movies, unseen_movies_ratings, unseen_movies_with_genres = get_unseen_movies_ratings_genres(movies_df,
                                                                                                       ratings_df,
                                                                                                       movies_with_genres,
                                                                                                       seen_movies)

    unseen_average_ratings = get_average_ratings(unseen_movies_with_genres, average_ratings_df)

    top_movies = get_top_movies(unseen_average_ratings)

    favourite_tags = get_favourite_tags(favourite_movies, tags_df)

    recommendations = get_movie_recommendations(top_movies, tags_df, favourite_tags, movies_df, 10)

    return recommendations


data_folder = "./processed_data"

ratings_df = pd.read_csv(data_folder + "/ratings.csv")
movies_df = pd.read_csv(data_folder + "/movies.csv")

expert_users_df = pd.read_csv(data_folder + "/expert_users.csv")
ratings_genres_df = pd.read_csv(data_folder + "/ratings_genres.csv")

genre_df = pd.read_csv(data_folder + "/genre.csv")
tags_df = pd.read_csv(data_folder + "/tags.csv")

average_ratings_df = pd.read_csv(data_folder + "/average_ratings.csv")


rec = content_recommender(expert_users_df, ratings_genres_df, movies_df, ratings_df, genre_df, tags_df,
                          average_ratings_df)
display(rec)


def test_system(num_users):
    k = 10

    precision_list = []

    for i in range(num_users):
        seen_movies, user = select_user_id(expert_users_df, ratings_genres_df)

        favourite_movies = get_favourite_movies(seen_movies)

        genre_set = get_genre_set(ratings_genres_df)

        genre_list = get_favourite_genres_powerset(genre_set, favourite_movies)

        movies_with_genres = get_movies_with_genres(genre_list, genre_df)

        user_ratings = ratings_df.loc[ratings_df['user'] == user]

        unseen_movies, unseen_movies_ratings, unseen_movies_with_genres = get_unseen_movies_ratings_genres(movies_df,
                                                                                                           ratings_df,
                                                                                                           movies_with_genres,
                                                                                                           seen_movies)

        genres_average_ratings = get_average_ratings(movies_with_genres, average_ratings_df)

        genres_average_ratings = genres_average_ratings.loc[genres_average_ratings['item'].isin(user_ratings.item)]

        top_movies = get_top_movies(genres_average_ratings)

        favourite_tags = get_favourite_tags(favourite_movies, tags_df)

        recommendations = get_movie_recommendations(top_movies, tags_df, favourite_tags, movies_df, 10)

        precision_list.append(precision_recall(recommendations, user))

    return np.mean(precision_list)


def precision_recall(recommendations, user):
    min_rating = 3

    user_ratings = ratings_df.loc[ratings_df['user'] == user]

    rec_ratings = user_ratings.loc[user_ratings['item'].isin(recommendations.item)]

    good_rec = rec_ratings.loc[rec_ratings['rating'] > min_rating]

    precision = len(good_rec) / len(recommendations)

    return precision


# precision = np.mean(test_system(10))
# print("precision = ")
# print(precision)
