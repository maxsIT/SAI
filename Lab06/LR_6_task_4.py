import argparse
import json
import numpy as np

# обробка вхідних аргументів
def build_arg_parser():
    parser = argparse.ArgumentParser(description = 'Compute similarity score')
    parser.add_argument('--user1', dest = 'user1', required = True, help = 'First user')
    parser.add_argument('--user2', dest = 'user2', required = True, help = 'Second user')
    parser.add_argument('--score-type', dest = 'score_type', required = True, choices = ['Euclidean', 'Pearson'], help = 'Similarity metric to be used')

    return parser

# метод обрахування евклідової оцінки
def euclidean_score(dataset, user1, user2):
    # перевірка чи користувачі існують у вхідному наборі даних
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')
    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # визначення набору фільмів, які були оцінені обома користувачами
    common_movies = {}
    for item in dataset[user1]:
        if item in dataset[user2]:  
            common_movies[item] = 1

    # обробка випадку, коли подібних фільмів немає
    if len(common_movies) == 0:
        return 0

    # пошук евклідової відстані між парами оцінок користувачів
    squared_diff = []
    for item in common_movies:
        squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))

    # обрахування евклідової відстані
    return 1 / (1 + np.sqrt(np.sum(squared_diff)))

def pearson_score(dataset, user1, user2):
    # перевірка чи користувачі існують у вхідному наборі даних
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')
    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # визначення набору фільмів, які були оцінені обома користувачами
    common_movies = {}
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    # обробка випадку, коли подібних фільмів немає
    num_rating = len(common_movies)
    if num_rating == 0:
        return 0

    # вираховування суми оцінок для всіх спільних фільмів
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])

    # вираховування квадратичної суми оцінок для всіх спільних фільмів
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])

    # вираховування суми добутків оцінок для всіх спільних фільмів
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])

    # вираховування параметрів оцінки подібності за Пірсоном 
    Sxy = sum_of_products - (user1_sum * user2_sum / num_rating)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_rating
    Syy = user2_squared_sum - np.square(user2_sum) / num_rating

    # перевірка випадку нульового добутку
    if Sxx * Syy == 0:
        return 0

    # вираховування оцінки подібності за Пірсоном
    return Sxy / np.sqrt(Sxx * Syy)

if __name__ == '__main__':
    # зчитування параметрів
    args = build_arg_parser().parse_args()
    user1 = args.user1
    user2 = args.user2
    score_type = args.score_type

    # зчитування файлу з рейтинговими оцінками
    rating_file = 'ratings.json'
    with open(rating_file, 'r') as f:
        data = json.loads(f.read())

    # обрахування подібності відповідно до обраної оцінки
    if score_type == 'Euclidean':
        print('\nEuclidean score:')
        print(euclidean_score(data, user1, user2))
    else:
        print('\nPearson score:')
        print(pearson_score(data, user1, user2))