#############################################################################################
# Project decription : Recommendation engine which selects best working model on the basis of
# Root mean square error on test dataset and generates movie recommnedations for set of users
##############################################################################################

import pandas as pd
import numpy as np
from scipy.sparse import linalg
from scipy import sparse
import os
from math import sqrt
from sklearn.metrics import pairwise_distances
from numpy.linalg import solve
from sklearn.metrics import mean_squared_error


# reference:https://www.ethanrosenthal.com/2016/01/09/explicit-matrix-factorization-sgd-als/


class CollabModel:
    """
    Class containing functions pertaining to collaborative filtering based on models like
    factorisation using alternative least squares and stochastic gradient descent
    """

    def __init__(self, ratings, num_factors, user_reg, item_reg, learning, user_bias, item_bias):
        self.ratings = ratings
        self.num_users, self.num_items = ratings.shape
        self.num_factors = num_factors
        self.item_reg = item_reg
        self.user_reg = user_reg
        self.learning = learning
        self.user_bias = user_bias
        self.item_bias = item_bias
        if self.learning == 'SGD':
            self.nonnullrows, self.nonnullcolumns = self.ratings.nonzero()
            self.num_samples = len(self.nonnullrows)

    def train(self, num_iteration=15, learning_rate=0.1):
        self.user_vec = np.random.normal(size=(self.num_users, self.num_factors))
        self.item_vec = np.random.normal(size=(self.num_items, self.num_factors))

        if self.learning == 'ALS':
            for i in range(num_iteration):
                self.user_vec = self.als_iteration_compute(self.user_vec, self.item_vec, self.ratings, self.user_reg,
                                                           'user')
                self.item_vec = self.als_iteration_compute(self.item_vec, self.user_vec, self.ratings, self.item_reg,
                                                           'item')

        elif self.learning == 'SGD':
            self.learning_rate = learning_rate
            self.user_bias = np.zeros(self.num_users)
            self.item_bias = np.zeros(self.num_items)
            self.global_bias = np.mean(self.ratings[self.ratings > 0])
            self.training_index = np.arange(self.num_samples)
            np.random.shuffle(self.training_index)
            self.sgd_iteration_compute()

    def sgd_iteration_compute(self):
        for ui in self.training_index:
            u = self.nonnullrows[ui]
            i = self.nonnullcolumns[ui]
            prediction = self.predict(u, i)
            e = (self.ratings[u, i] - prediction)

            self.user_bias[u] = self.user_bias[u] + self.learning_rate * (e - self.user_reg * self.user_bias[u])
            self.item_bias[i] = self.item_bias[i] + self.learning_rate * (e - self.item_reg * self.item_bias[i])

            # Update latent factors
            self.user_vec[u, :] += self.learning_rate * (e * self.item_vec[i, :] - self.user_reg * self.user_vec[u, :])
            self.item_vec[i, :] += self.learning_rate * (e * self.user_vec[u, :] - self.item_reg * self.item_vec[i, :])

    def als_iteration_compute(self, latent_vec, fixed_vec, ratings, lamd, method='user'):
        fix_vec_T_fix_vec = np.dot(fixed_vec.T, fixed_vec)

        if method == 'user':
            lambda_iden = sparse.eye(fix_vec_T_fix_vec.shape[0]) * lamd
            for u in range(self.num_users):
                latent_vec[u, :] = solve((fix_vec_T_fix_vec + lambda_iden),
                                         ratings[u, :].dot(fixed_vec))
        elif method == 'item':
            lambda_iden = sparse.eye(fix_vec_T_fix_vec.shape[0]) * lamd
            for i in range(self.num_items):
                latent_vec[i, :] = solve((fix_vec_T_fix_vec + lambda_iden),
                                         ratings[:, i].dot(fixed_vec))

        return latent_vec

    def predict(self, u, i):
        return self.user_vec[u, :].dot(self.item_vec[i, :].T)

    def predict_full(self):
        predictions = np.zeros((self.num_users, self.num_items))
        for u in range(self.num_users):
            for i in range(self.num_items):
                predictions[u, i] = self.predict(u, i)
        return predictions


class CollabMemory:
    """
        Class containing functions pertaining to collaborative filtering based on memory or similarity based like
        User-User / Item-Item
    """

    def __init__(self, ratings):
        self.ratings = ratings
        self.num_users, self.num_items = ratings.shape

    def distance_based(self, ratings_matrix, method='item', metric='cosine'):
        if method == 'item':
            movie_similarity = 1 - pairwise_distances(ratings_matrix, metric=metric)
            np.fill_diagonal(movie_similarity, 0)  # Filling diagonals with 0s for future use when sorting is done
            return movie_similarity
        else:
            user_similarity = 1 - pairwise_distances(ratings_matrix, metric=metric)
            np.fill_diagonal(user_similarity, 0)  # Filling diagonals with 0s for future use when sorting is done
            return user_similarity

    def predict_full(self, ratings, similarity, method='user', nobias=True):
        if method == 'user':
            if nobias:
                user_bias = ratings.mean(axis=1)
                ratings = (ratings - user_bias[:, np.newaxis]).copy()
            pred = similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
            if nobias:
                pred += user_bias[:, np.newaxis]
        elif method == 'item':
            if nobias:
                item_bias = ratings.mean(axis=0)
                ratings = (ratings - item_bias[np.newaxis, :]).copy()
            pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
            if nobias:
                pred += item_bias[np.newaxis, :]
        return pred

    def predict_full_topk(self, ratings, similarity, method='user', k=40, nobias=True):
        pred = np.zeros(ratings.shape)
        if method == 'user':
            if nobias:
                user_bias = ratings.mean(axis=1)
                ratings = (ratings - user_bias[:, np.newaxis]).copy()
            for i in range(ratings.shape[0]):
                # print(ratings.shape[0])
                # print(similarity.shape)
                # print(i)
                # print(similarity[:,i])
                top_k_users = [np.argsort(similarity[:, i])[:-k - 1:-1]]
                for j in range(ratings.shape[1]):
                    pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                    pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
            if nobias:
                pred += user_bias[:, np.newaxis]
        if method == 'item':
            if nobias:
                item_bias = ratings.mean(axis=0)
                ratings = (ratings - item_bias[np.newaxis, :]).copy()
            for j in range(ratings.shape[1]):
                top_k_items = [np.argsort(similarity[:, j])[:-k - 1:-1]]
                for i in range(ratings.shape[0]):
                    pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
                    pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))
            if nobias:
                pred += item_bias[np.newaxis, :]
        return pred


def train_test_split_rem(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        if len(ratings[user, :].nonzero()[0]) >= 20:
            test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                            size=10,
                                            replace=False)
            train[user, test_ratings] = 0.
            test[user, test_ratings] = ratings[user, test_ratings]

    # Test and training are truly disjoint
    assert (np.all((train * test) == 0))
    return train, test


def MF(train):
    user_ratings_mean = np.mean(train, axis=1)
    Ratings_nobias = train - user_ratings_mean.reshape(-1, 1)
    u, s, vt = linalg.svds(Ratings_nobias, k=20)  # tweak k, dimensionality for rank matrix
    s_diag_matrix = np.diag(s)
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt) + user_ratings_mean.reshape(-1, 1)
    return X_pred


def create_dict(a):
    val_to_ix = {ch: i for i, ch in enumerate(sorted(a))}
    return val_to_ix


def get_dict(a):
    ix_to_val = {i: ch for i, ch in enumerate(sorted(a))}
    return ix_to_val


def root_mean_square_error(prediction, actual):
    prediction = prediction[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, actual))


def get_similar_movie(user_inp, rating_matrix, movies):
    inp = movies[movies['title'] == user_inp].movieId.tolist()
    print(inp)
    inp = inp[0]
    Imc = CollabMemory(rating_matrix)
    movie_similarity = Imc.distance_based(ratings_matrix.T, method='item', metric='correlation')
    movies['similarity'] = movie_similarity[inp]
    return movies.loc[:, ['title', 'genres', 'similarity']].sort_values(["similarity"], ascending=False)[1:10]


def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    # Get and sort the user's predictions
    user_row_number = userID - 1  # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.userId == (userID)]
    # user_full = (user_data.merge(movies_df, how='left', left_on='movieId', right_on='movieId').
    #              sort_values(['rating'], ascending=False)
    #              )
    user_full = user_data.sort_values(['rating'], ascending=False)

    print('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
    print('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))

    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])].
                           merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                                 left_on='movieId',
                                 right_on='movieId').
                           rename(columns={user_row_number: 'Predictions'}).
                           sort_values('Predictions', ascending=False).
                           iloc[:num_recommendations, :-1]
                           )
    return user_full, recommendations


def get_bestmodel_pred(rating_matrix, verbose=0):

    # We will split our data into training and test sets by removing 5 ratings per user from the training set and placing them in the test set.
    train, test = train_test_split_rem(ratings_matrix.values)
    # Predict ratings on the training data with user and item similarity score
    k = 0
    rmse_dict = {}
    Imc = CollabMemory(train)
    while k < 1:
        for dist in ['correlation', 'cosine', 'jaccard']:
            user_simil = Imc.distance_based(train, method='user', metric='correlation')

            user_prediction = Imc.predict_full(train, user_simil, method='user', nobias=False)
            # RMSE on the train data
            if verbose > 0:
                print('User-based CF train bias RMSE: ' + str(root_mean_square_error(user_prediction, train)))
                # RMSE on the test data
                print('User-based CF test bias RMSE: ' + str(root_mean_square_error(user_prediction, test)))
            rmse_dict['user-' + dist + '-bias'] = root_mean_square_error(user_prediction, test)
            user_prediction = Imc.predict_full_topk(train, user_simil, method='user', nobias=False)
            if verbose > 0:
                # RMSE on the train data
                print('User-based CF train bias top k RMSE: ' + str(root_mean_square_error(user_prediction, train)))
                # RMSE on the test data
                print('User-based CF test bias top k RMSE: ' + str(root_mean_square_error(user_prediction, test)))
            rmse_dict['user-' + dist + '-bias-top40'] = root_mean_square_error(user_prediction, test)

            user_prediction = Imc.predict_full(train, user_simil, method='user', nobias=True)
            if verbose > 0:
                # RMSE on the train data
                print('User-based CF train nobias RMSE: ' + str(root_mean_square_error(user_prediction, train)))
                # RMSE on the test data
                print('User-based CF test nobias RMSE: ' + str(root_mean_square_error(user_prediction, test)))
            rmse_dict['user-' + dist + '-nobias'] = root_mean_square_error(user_prediction, test)

            user_prediction = Imc.predict_full_topk(train, user_simil, method='user', nobias=True)
            if verbose > 0:
                # RMSE on the train data
                print('User-based CF train nobias topk RMSE: ' + str(root_mean_square_error(user_prediction, train)))
                # RMSE on the test data
                print('User-based CF test nobias top k RMSE: ' + str(root_mean_square_error(user_prediction, test)))
            rmse_dict['user-' + dist + '-nobias-top40'] = root_mean_square_error(user_prediction, test)

            item_simil = Imc.distance_based(train.T, method='item', metric='correlation')

            item_prediction = Imc.predict_full(train, item_simil, method='item', nobias=False)
            if verbose > 0:
                # RMSE on the train data
                print('item-based CF train bias RMSE: ' + str(root_mean_square_error(item_prediction, train)))
                # RMSE on the test data
                print('item-based CF test bias RMSE: ' + str(root_mean_square_error(item_prediction, test)))
            rmse_dict['item-' + dist + '-bias'] = root_mean_square_error(item_prediction, test)

            item_prediction = Imc.predict_full_topk(train, item_simil, method='item', nobias=False)
            if verbose > 0:
                # RMSE on the train data
                print('item-based CF train bias top k RMSE: ' + str(root_mean_square_error(item_prediction, train)))
                # RMSE on the test data
                print('item-based CF test bias top k RMSE: ' + str(root_mean_square_error(item_prediction, test)))
            rmse_dict['item-' + dist + '-bias-top40'] = root_mean_square_error(item_prediction, test)

            item_prediction = Imc.predict_full(train, item_simil, method='item', nobias=True)
            if verbose > 0:
                # RMSE on the train data
                print('item-based CF train nobias RMSE: ' + str(root_mean_square_error(item_prediction, train)))
                # RMSE on the test data
                print('item-based CF test nobias RMSE: ' + str(root_mean_square_error(item_prediction, test)))
            rmse_dict['item-' + dist + '-nobias'] = root_mean_square_error(item_prediction, test)

            item_prediction = Imc.predict_full_topk(train, item_simil, method='item', nobias=True)
            if verbose > 0:
                # RMSE on the train data
                print('item-based CF train nobias topk RMSE: ' + str(root_mean_square_error(item_prediction, train)))
                # RMSE on the test data
                print('item-based CF test nobias top k RMSE: ' + str(root_mean_square_error(item_prediction, test)))
            rmse_dict['item-' + dist + '-nobias-top40'] = root_mean_square_error(item_prediction, test)

        X_pred = MF(train)
        if verbose > 0:
            print('matrix-factorization CF RMSE: %.2f' % root_mean_square_error(X_pred, test))
        rmse_dict['MF-nobias'] = root_mean_square_error(item_prediction, test)

        MF_ALS = CollabModel(train, num_factors=10, learning='ALS', user_reg=0, item_reg=0, user_bias=0, item_bias=0)
        MF_ALS.train(20)
        if verbose > 0:
            print('RMSE: ' + str(root_mean_square_error(MF_ALS.predict_full(), test)))
        rmse_dict['ALS'] = root_mean_square_error(MF_ALS.predict_full(), test)

        MF_SGD = CollabModel(train, num_factors=10, learning='SGD', user_reg=0, item_reg=0, user_bias=0, item_bias=0)
        print(MF_SGD.num_samples)
        MF_SGD.train(20, learning_rate=0.001)
        if verbose > 0:
            print('RMSE: ' + str(root_mean_square_error(MF_SGD.predict_full(), test)))
        rmse_dict['SGD'] = root_mean_square_error(MF_SGD.predict_full(), test)

        k = 1

    print(min(rmse_dict, key=rmse_dict.get))
    bestalgo = min(rmse_dict, key=rmse_dict.get)
    if bestalgo.split('-')[0] == 'user':
        simil = Imc.distance_based(ratings_matrix.values, method=bestalgo.split('-')[0],
                                   metric=bestalgo.split('-')[1])
    else:
        simil = Imc.distance_based(ratings_matrix.T.as_matrix(), method=bestalgo.split('-')[0],
                                   metric=bestalgo.split('-')[1])
    if bestalgo == 'MF-nobias':
        X_pred = MF(ratings_matrix.as_matrix())
    elif len(bestalgo.split('-')) > 2 and bestalgo.split('-')[2] == 'bias':
        if len(bestalgo.split('-')) > 3 and bestalgo.split('-')[3] == 'top40':
            X_pred = Imc.predict_full_topk(ratings_matrix.values, simil, method=bestalgo.split('-')[0],
                                           nobias=False)
        else:
            X_pred = Imc.predict_full(ratings_matrix.values, simil, method=bestalgo.split('-')[0], nobias=False)

    elif len(bestalgo.split('-')) > 2 and bestalgo.split('-')[2] == 'nobias':
        if len(bestalgo.split('-')) > 3 and bestalgo.split('-')[3] == 'top40':
            X_pred = Imc.predict_full_topk(ratings_matrix.values, simil, method=bestalgo.split('-')[0],
                                           nobias=True)
        else:
            X_pred = Imc.predict_full(ratings_matrix.values, simil, method=bestalgo.split('-')[0], nobias=True)
    elif bestalgo == 'ALS':
        X_pred = MF_ALS.predict_full()
    elif bestalgo == 'SGD':
        X_pred = MF_SGD.predict_full()
    else:
        print('algo doesnot exist')

    return X_pred


if __name__ == '__main__':
    verbose = 0
    path = os.path.join('D:', '\Learning', 'Recommendation_Engine', 'data', 'movielens', 'ml-20m')
    # movies = pd.read_csv(os.path.join(path, 'movies.csv'))
    # rating = pd.read_csv(os.path.join(path, 'ratings.csv'))
    # movie_data = pd.merge(rating, movies, on='movieId')
    # movie_data.to_csv(os.path.join(path,'Moviesdata.csv'),index=False)
    movie_data = pd.read_csv(os.path.join(path, 'Moviesdata.csv'))
    movie_list = movie_data['movieId'].unique()[:1000]
    user_list = movie_data['userId'].unique()[:10000]

    movie_data['rating'] = movie_data['rating'].astype(float)
    movie_data = movie_data[movie_data['movieId'].isin(movie_list)]
    movie_data = movie_data[movie_data['userId'].isin(user_list)]
    movie_dict = create_dict(movie_list)
    movie_data['movieId'] = movie_data['movieId'].map(movie_dict)
    movies = movie_data.loc[:, ['movieId', 'title', 'genres']].drop_duplicates()

    ratings_matrix = movie_data.pivot_table(index=['userId'], columns=['movieId'], values='rating').reset_index(drop=True)
    ratings_matrix.fillna(0, inplace=True)
    print(ratings_matrix.shape)

    X_pred = get_bestmodel_pred(ratings_matrix, verbose)

    preds = pd.DataFrame(X_pred, columns=ratings_matrix.columns)

    print("Predictions for users based on their user id")
    for id in [1, 15, 200, 34, 1310]:
        try:
            already_rated, predictions = recommend_movies(preds, id, movies, movie_data, 20)
            print(already_rated.columns)
            print(already_rated.loc[:, ['title', 'genres']].head(20))
            print(predictions.loc[:, ['title', 'genres']])
        except Exception as e:
            print("error ", e)
            continue

    try:
        # user_inp = input('Enter one of your favourite movie title based on which recommendations are to be made: ')
        user_inp = "Shawshank Redemption, The (1994)"

        movie_list = get_similar_movie(user_inp, ratings_matrix, movies)
        print("Recommended movies based on your choice of ", user_inp, ": \n", movie_list)
    except Exception as e:
        print(e)
