import argparse
import re
import os
import csv
import math
import collections as coll
import time


def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args 

def parse_file(filename):
    """
    user_ratings[user_id] = {movie_id: rating}
    movie_ratings[movie_id] = {user_id: rating}
    """
    user_ratings = {}
    movie_ratings = {}
    with open(filename) as txtfile:
        content = csv.reader(txtfile)
        list_of_lists = [[int(line) if index < 2 else float(line) for index, line in enumerate(ls)] for ls in content]
        for row in list_of_lists:
            movie_id = row[0]
            user_id = row[1]
            rating = row[2]
            # if key in dictionary
            if user_id in user_ratings: # check the key in dictionary or not
                user_ratings[user_id][movie_id] = rating
            else:                   # otherwise assign the value to the key
                user_ratings[user_id] = {movie_id : rating}
            if movie_id in movie_ratings:
                movie_ratings[movie_id][user_id] = rating
            else:
                movie_ratings[movie_id] = {user_id : rating}
        return user_ratings, movie_ratings


def compute_average_user_ratings(user_ratings):
    """ 
    Given a the user_rating dict compute average user ratings
    Input: user_ratings (dictionary of user, movies, ratings)
    Output: ave_ratings (dictionary of user and ave_ratings)
    """
    ave_ratings = {}
    for user_id in user_ratings:
        ratings = user_ratings[user_id].values()
        avg = sum(ratings) / len(ratings)
        ave_ratings[user_id] = avg
    return ave_ratings # average ratings


def compute_user_similarity(d1, d2, ave_rat1, ave_rat2):
    """ Computes similarity between two users
        Input: d1, d2, (dictionary of user ratings per user) 
        ave_rat1, ave_rat2 average rating per user (float)
        Ouput: user similarity (float)
    """
    items_in_common = set(d1.keys()) & set(d2.keys())
    tmp1 = 0.0
    tmp2 = 0.0
    tmp3 = 0.0
    w_12 = 0.0
    for i in items_in_common:
        tmp1 += (d1[i] - ave_rat1) * (d2[i] - ave_rat2)
        tmp2 += (d1[i] - ave_rat1) ** 2
        tmp3 += (d2[i] - ave_rat2) ** 2
    if tmp2 * tmp3 != 0:
        w_12 = tmp1 / math.sqrt(tmp2 * tmp3)
    else:
        w_12 = 0
    return w_12

# def compute_user_similarity(d1, d2, ave_rat1, ave_rat2):
#     """ Computes similarity between two users
#         Input: d1, d2, (dictionary of user ratings per user) 
#         ave_rat1, ave_rat2 average rating per user (float)
#         Ouput: user similarity (float)
#     """
#     items_in_common = set(d1.keys()) & set(d2.keys())
#     w_12 = 0.0
#     l1 = [d1[i] - ave_rat1 for i in items_in_common]
#     l2 = [d2[i] - ave_rat2 for i in items_in_common]
#     tmp1 = sum(x * y for x, y in zip(l1, l2))
#     tmp2 = sum(x * y for x, y in zip(l1, l1))
#     tmp3 = sum(x * y for x, y in zip(l2, l2))
#     if tmp2 * tmp3 != 0:
#     	w_12 = tmp1 / (tmp2 * tmp3) ** 0.5
#     else:
# 		w_12 = 0
#     return w_12


def main():
    """
    This function is called from the command line via
    python cf.py --train [path to filename] --test [path to filename]
    """
    args = parse_argument()
    train_file = args['train'][0]
    test_file = args['test'][0]
    print train_file, test_file
    user_ratings, movie_ratings = parse_file(train_file)
    avg_ratings = compute_average_user_ratings(user_ratings)
    n = 0
    count = 0
    with open(test_file) as f:
    	content = csv.reader(f)
    	count = sum([1 for row in content])
    with open(test_file) as f:
    	content = csv.reader(f)
    	output = [None] * count
    	diff_square = 0.0
        abs_diff = 0.0
        list_of_lists = [[int(line) if index < 2 else float(line) for index, line in enumerate(ls)] for ls in content]
        for row in list_of_lists:
            movie_k = row[0]
            user_i = row[1]
            avg_i = avg_ratings[user_i]
            # W : sum of |w|  d 
            W = 0.0
            d = 0.0 
            pred = 0.0
            d1 = user_ratings[user_i]
            ave_rat1 = avg_ratings[user_i]
            # all the user who rates movie_k before
            for user in movie_ratings[movie_k].keys():
                d2 = user_ratings[user]
                ave_rat2 = avg_ratings[user]
                w = compute_user_similarity(d1, d2, ave_rat1, ave_rat2)
                W += abs(w)
                R_j_bar = avg_ratings[user]
                d += w * (user_ratings[user][movie_k] - R_j_bar)
                #w could be 0
            if W != 0:
                pred = ave_rat1 + 1/W * d
            else:
                pred = ave_rat1
            row.append(pred)
            diff = row[2] - row[3]
            diff_square += diff ** 2
            abs_diff += abs(diff)
            output[n] = row
            n += 1
        rmse = round(math.sqrt(diff_square / float(n)), 4)
        mae = round(abs_diff / float(n), 4)
        print rmse, mae
    f_p = open('predictions.txt', 'w')
    w = csv.writer(f_p)
    w.writerows(output)
    f_p.close()
    
if __name__ == '__main__':
    main()






