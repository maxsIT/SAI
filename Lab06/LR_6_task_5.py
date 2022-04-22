import argparse
import json
import numpy as np

# обробка вхідних аргументів
def build_arg_parser():
    parser = argparse.ArgumentParser(description = 'Find users who are similar to the input user')
    parser.add_argument('--user', dest = 'user', required = True, help = 'Input user')

    return parser

# вирахування оцінки подібності за Пірсоном
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

# метод пошуку подібних користувачів
def find_similar_users(dataset, user, num_users):
    # перевірка чи користувач існує у наборі даних
    if user not in dataset:
        raise TypeError('Cannot find ' + user + ' in the dataset')

    # вирахування подібності оцінок з всіма іншими користувачами
    scores = np.array([[x, pearson_score(dataset, user, x)] for x in dataset if x != user])
    # сортування оцінок подібності за спаданням
    scores_sorted = np.argsort(scores[:, 1])[::-1]
    # вибір num_users найбільш подібних користувачів
    top_users = scores_sorted[:num_users]
    
    return scores[top_users]

if __name__ == '__main__':
    # зчитування параметрів
    args = build_arg_parser().parse_args()
    user = args.user

    # зчитування файлу з рейтинговими оцінками
    rating_file = 'ratings.json'
    with open(rating_file, 'r') as f:
        data = json.loads(f.read())

    print('\nUsers similar to ', user + ':\n')
    # пошук подібних користувачів
    similar_users = find_similar_users(data, user, 3)
    print('User\t\t\tSimilarity score')
    print('-' * 41)
    # виведення імен та оцінок подібності найбільш схожих користувачів
    for item in similar_users:
        print(item[0], '\t\t', round(float(item[1]), 2))