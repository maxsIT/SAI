import argparse
import json
import numpy as np

# обробка вхідних аргументів
def build_arg_parser():
    parser = argparse.ArgumentParser(description = 'Find the movie recommendations for the given user')
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

def get_recommendations(dataset, input_user):
    # перевірка чи користувач існує у наборі даних
    if input_user not in dataset:
        raise TypeError('Cannot find ' + user + ' in the dataset')

    overall_scores = {}
    similarity_scores = {}

    # аналіз користувачів, окрім користувача для якого ми шукаємо рекомендації
    for user in [x for x in dataset if x != input_user]:
        # вирахування оцінки подібності за Пірсоном
        similarity_score = pearson_score(dataset, input_user, user)
        # якщо оцінка подібності <= 0, користувачі не є подібними та їх рекомендації не будуть корисними
        if similarity_score <= 0:
            continue

        # відфільтровуємо список фільмів переглянутих поточним користувачем, які ще не дивився користувач для якого ми шукаємо рекомендації
        filtered_list = [x for x in dataset[user] if x not in dataset[input_user] or dataset[input_user][x] == 0]
        for item in filtered_list:
            # вираховуємо зважену оцінку фільму та зберігаємо цю оцінку та оцінку подібності для подалюшого використання
            overall_scores.update({item: dataset[user][item] * similarity_score})
            similarity_scores.update({item: similarity_score})

        # якщо не було знайдено фільмів, які дивився тільки поточний користувач, виводимо повідомлення про те, що ми не можемо надати рекомендації
        if len(overall_scores) == 0:
            return ['No recommendations possible']

        # конвертуэмо зважену оцінку в оцінку, яка була поставлена поточним користувачем
        movie_scores = np.array([[score / similarity_scores[item], item] for item, score in overall_scores.items()])
        # сортуємо фільми за спаданням їх оцінки
        movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]
        # формуємо список рекомендцій
        movie_recommendations = [movie for _, movie in movie_scores]

        return movie_recommendations
        

if __name__ == '__main__':
    # зчитування параметрів
    args = build_arg_parser().parse_args()
    user = args.user

    # зчитування файлу з рейтинговими оцінками
    rating_file = 'ratings.json'
    with open(rating_file, 'r') as f:
        data = json.loads(f.read())

    print('\nMovie recommendations for ', user + ':')
    # отримання та виведення списку рекомендації
    movies = get_recommendations(data, user)
    for i, movie in enumerate(movies):
        print(str(i + 1) + '.', movie)