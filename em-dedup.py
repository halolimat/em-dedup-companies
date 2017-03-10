#! python3

import numpy as np
import os
import csv
import string
import pickle
import affinegap
from functools import lru_cache
from collections import defaultdict
import copy


#
# Implementation of the core expectation-maximization algorithm proposed in the following paper:
# Bohannon, P., Dalvi, N., Raghavan, M., & Olteanu, M. (2014). Deduplicating a Places Database. In WWW.
#
# The idea, is that we learn 2 distributions over our vocabulary, one for the set of words that
# are core to each company name, and another for the words that are in the background. Then we
# can use these distributions to compare strings more appropriately, since we only care about
# matching "core words"
#


# file name for where we save the models:
FILE_CORE = 'core.pickle'
FILE_BACK = 'back.pickle'

EPS = 1e-25
EPS2 = 1e-50
UNIQUE_NAMES = []


def read_data(file_name, use_lower=True):
    exclude = set(string.punctuation)

    data = []
    word_freq = defaultdict(int)
    with open(file_name, 'r') as fin:
        csv_reader = csv.reader(fin)
        # skip header
        next(csv_reader)

        for i, row in enumerate(csv_reader):
            # grab the company name
            line = row[-1].strip()

            if use_lower:
                line = line.lower()

            # remove punctuation
            line = ''.join(ch for ch in line if ch not in exclude)

            # TODO: may need to swap in a more sophisticated tokenizer here
            words = line.split(' ')

            data.append(words)
            # build a unique identifier so we can debug if necessary
            uid = unique_company_name(line, i)
            UNIQUE_NAMES.append(uid)

            for w in words:
                word_freq[w] += 1

    return data, word_freq


def initialize_z(data):
    """
    Build the initial data structure for the binary variable z that determines which words
    from each company name are part of the core
    :param data:
    :return:
    """
    z = {}
    for i, row in enumerate(data):
        uid = UNIQUE_NAMES[i]
        z[uid] = {}

        for w in row:
            z[uid][w] = 1
    return z


def init_probabilities(word_freq):

    """
    Initialize the core and background probabilities over the vocabulary to be the uniform distribution
    :param word_freq:
    :return:
    """
    prob_core = {}
    prob_back = {}

    total_core = 0.0
    total_back = 0.0
    for word, freq in word_freq.items():
        prob_core[word] = 1
        prob_back[word] = 1

        total_core += 1
        total_back += 1

    for w, v in prob_core.items():
        prob_core[w] /= total_core
        prob_back[w] /= total_back

    return prob_core, prob_back


def unique_company_name(company, index):
    # build unique string identifier for this company
    if type(company) is list:
        c = ' '.join(company)
    else:
        c = company
    return '{}_{}'.format(c, index)


def log_likelihood(data, prob_core, prob_back, z):

    llike = 0.0
    for i, company in enumerate(data):
        uid = unique_company_name(company, i)
        for word in company:
            p_core = prob_core[word]
            p_back = prob_back[word]

            # just checking here - this really shouldn't happen though!
            if p_core == 0.0:
                p_core = EPS
            if p_back == 0.0:
                p_back = EPS

            llike += z[uid][word] * np.log(p_core) + (1.0 - z[uid][word]) * np.log(p_back)
    return llike


def compute_z_denominator(company, prob_core, prob_back):
    # sum the relative weight of core probability to background probability of all words in the company name
    denom = 0.0
    for word in company:
        denom += (prob_core[word]) / (prob_back[word])

    return denom


def compute_prob_numerators(data, z_current, word):
    # compute the numerator of the expectation of z
    c_num = EPS
    b_num = EPS

    # for each company name
    for i, company_a in enumerate(data):
        uid_a = UNIQUE_NAMES[i]
        # if the given word is in the name...
        if word in company_a:
            # we sum the z value (of this word in this company name)
            c_num += z_current[uid_a][word]
            # and the (1 - z) value
            b_num += 1.0 - z_current[uid_a][word]

    return c_num, b_num


def compute_prob_denominators(data, vocabulary, z_current):
    c_den = EPS
    b_den = EPS

    # for each word in the vocabulary
    for word_a in vocabulary.keys():
        # for each company name...
        for i, company_a in enumerate(data):
            uid_a = UNIQUE_NAMES[i]
            # if the word exists in this company name
            if word_a in company_a:
                z = z_current[uid_a][word_a]

                # we sum the z value
                c_den += z
                b_den += 1.0 - z

    return c_den, b_den


def _normalize(prob):
    """
    Given a probability distribution (dict), ensure that the values sum to 1
    :param prob:
    :return:
    """
    total = 0.0

    for k, v in prob.items():
        if v < 0:
            v = EPS
            prob[k] = v
        total += v

    for k in prob.keys():
        prob[k] /= total

    return prob


def expectation_maximization(data, vocabulary, prob_core, prob_back, z, max_iters=50):
    """
    Core expectation-maximization algorithm for learning the core and background word
    probabilities given all of the company names

    :param data:
    :param vocabulary:
    :param prob_core:
    :param prob_back:
    :param z:
    :param max_iters:
    :return:
    """

    for iter in range(max_iters):
        z_current = copy.deepcopy(z)
        prob_core_current = copy.deepcopy(prob_core)
        prob_back_current = copy.deepcopy(prob_back)

        # compute denominator for core & background probabilities
        # we do this outside the loops since this is only dependent on the current values of z
        c_den, b_den = compute_prob_denominators(data, vocabulary, z_current)
        for i, company in enumerate(data):
            uid = unique_company_name(company, i)

            z_den = compute_z_denominator(company, prob_core_current, prob_back_current)
            for word in company:
                #
                # expectation step
                #

                # update z
                z_num = prob_core_current[word] / prob_back_current[word]
                z_current[uid][word] = z_num / z_den

                #
                # maximization step
                #

                # update core & background probability distributions
                c_num, b_num = compute_prob_numerators(data, z_current, word)

                p_core = c_num / c_den
                p_back = b_num / b_den

                # just here as a safety mechanism
                if p_core < 0:
                    # import pdb; pdb.set_trace()
                    p_core = EPS2
                if p_back < 0:
                    # import pdb; pdb.set_trace()
                    p_back = EPS2

                prob_core_current[word] = p_core
                prob_back_current[word] = p_back

        z = z_current
        # prob_core = prob_core_current
        # prob_back = prob_back_current

        prob_core = _normalize(prob_core_current)
        prob_back = _normalize(prob_back_current)
        # print('---- Z ----')
        # print(z)
        # print(' ---- CORE -----')
        # print(prob_core)
        # print(' ---- BACKGROUND -----')
        # print(prob_back)
        # print('\n')
        llike = log_likelihood(data, prob_core, prob_back, z)
        print("Iteration = {:4}\tLog Likelihood = {:.4}".format(iter, llike))

    prob_core = _normalize(prob_core)
    prob_back = _normalize(prob_back)
    return prob_core, prob_back


def get_core_probability(word, prob_core, prob_back, alpha=0.25):
    pc = prob_core.get(word, EPS2)
    pb = prob_back.get(word, EPS2)

    prob = alpha * pc / (alpha * pc + (1 - alpha) * pb)
    return prob


@lru_cache(32)
def get_key(c1, c2):
    dist_key = '{}_{}'.format(c1, c2)
    return dist_key


def edit_distances(company1, company2):
    both, _, _ = get_word_sets(company1, company2)
    dist = {}

    for c1 in company1:
        if c1 in both:
            continue

        for c2 in company2:
            if c2 in both:
                continue

            dist_key = get_key(c1, c2)
            dist[dist_key] = affinegap.normalizedAffineGapDistance(c1, c2, matchWeight=0)

    return dist


def apw_prob(edit_dist):
    return 1. / (1 + np.exp(edit_dist))


def get_all_possible_worlds(company1, company2):
    # approximate possible worlds for the 2 company names used to score the likelihood of a match
    # we do this by looking at:
    # 1. the original string (assume no mistakes were made)
    # 2. fully transform company1 to match company2
    # 3. transforming each token that doesn't exist in both company names
    # all of these are weighted approximately inversely to the edit distance required

    # TODO this method needs to be modified to implement true possible worlds semantics
    distances = edit_distances(company1, company2)
    total_prob = 0.0

    # original
    apws = [(apw_prob(0), company1, company2)]
    total_prob += apw_prob(0)
    # full match
    full_match_dist = affinegap.normalizedAffineGapDistance(' '.join(company1), ' '.join(company2), matchWeight=0)
    total_prob += apw_prob(full_match_dist)
    apws.append((apw_prob(full_match_dist), company2[:], company2[:]))

    for k, dist in distances.items():
        if dist > 4:
            continue
        w1, w2 = k.split('_')

        # update company1
        c1 = company1[:]
        idx = c1.index(w1)
        c1[idx] = w2

        apws.append((apw_prob(dist), c1, company2[:]))
        total_prob += apw_prob(dist)

    # normalize APW distribution
    apws_norm = []
    for prob, c1, c2 in apws:
        apws_norm.append((prob / total_prob, c1, c2))

    return apws_norm


def get_word_sets(company1, company2):
    """
    Given 2 company names (lists), compute the set of words common to both,
    words just in company1 and words just in company2
    :param company1:
    :param company2:
    :return:
    """
    sc1 = set(company1)
    sc2 = set(company2)

    # words in c1, not in c2
    word_c1_only = sc1 - sc2

    # words in c2, not in c1
    word_c2_only = sc2 - sc1

    # words in both
    word_in_both = sc1.intersection(sc2)

    return word_in_both, word_c1_only, word_c2_only


def prob_match(company1, company2, prob_core, prob_back):
    words_in_both, words_c1_only, words_c2_only = get_word_sets(company1, company2)

    prob = 1.0
    for w in words_c1_only:
        prob *= (1. - get_core_probability(w, prob_core, prob_back))

    for w in words_c2_only:
        prob *= (1. - get_core_probability(w, prob_core, prob_back))

    for w in words_in_both:
        p = get_core_probability(w, prob_core, prob_back)
        prob *= (p ** 2 + (1. - p) ** 2)

    return prob


def apw_match_prob(company1, company2, prob_core, prob_back):
    total = 0.0
    for prob_w, c1, c2 in get_all_possible_worlds(company1, company2):
        prob = prob_w * prob_match(c1, c2, prob_core, prob_back)
        total += prob

    return total


def write(prob_core, prob_back):
    with open(FILE_CORE, 'wb') as handle:
        pickle.dump(prob_core, handle)

    with open(FILE_BACK, 'wb') as handle:
        pickle.dump(prob_back, handle)


def load_probabilities():
    if not os.path.isfile(FILE_CORE) or not os.path.isfile(FILE_BACK):
        return None, None

    print('Loading probability distributions from disk...')
    with open(FILE_CORE, 'rb') as handle:
        prob_core = pickle.load(handle)

    with open(FILE_BACK, 'rb') as handle:
        prob_back = pickle.load(handle)

    return prob_core, prob_back


def test(c1, c2, prob_core, prob_back):
    prob = prob_match(c1, c2, prob_core, prob_back)
    print('Company1: {}\tCompany2: {}'.format(c1, c2))
    print('Probability: {}'.format(prob))
    print('------')


def run_examples(prob_core, prob_back):
    # different companies
    company1 = ['peets', 'coffee']
    company2 = ['starbucks', 'coffee']

    pp = affinegap.normalizedAffineGapDistance(' '.join(company1), ' '.join(company2), matchWeight=0)
    print('edit distance: {}'.format(pp))
    p = apw_match_prob(company1, company2, prob_core, prob_back)
    print(company1, company2, p)
    # test(company1, company2, prob_core, prob_back)

    company1 = ['starbuck', 'company']
    company2 = ['starbucks', 'coffee']
    pp = affinegap.normalizedAffineGapDistance(' '.join(company1), ' '.join(company2), matchWeight=0)
    print('edit distance: {}'.format(pp))

    p = apw_match_prob(company1, company2, prob_core, prob_back)
    print(company1, company2, p)
    # test(company1, company2, prob_core, prob_back)

    company1 = ['starbucks', 'coffe']
    company2 = ['starbucks', 'coffee']
    p = apw_match_prob(company1, company2, prob_core, prob_back)
    print(company1, company2, p)
    # test(company1, company2, prob_core, prob_back)

    company1 = ['starbucks']
    company2 = ['starbucks', 'coffee', 'company']
    p = apw_match_prob(company1, company2, prob_core, prob_back)
    print(company1, company2, p)
    # test(company1, company2, prob_core, prob_back)

    company1 = ['coffee', 'company']
    company2 = ['starbucks', 'coffee', 'company']
    p = apw_match_prob(company1, company2, prob_core, prob_back)
    print(company1, company2, p)
    # test(company1, company2, prob_core, prob_back)

    for w in ['starbucks', 'peets', 'coffee', 'company', 'incorporated', 'inc', 'corporation', 'llc']:
        print('{:20}\t{:.4}\t{:.4}'.format(w, prob_core[w], prob_back[w]))

def run_test(data, prob_core, prob_back):

    for x in data:
        for y in x:
            if prob_core[y] > prob_back[y]:
                print (y, 1)
            else:
                print (y, 0)

        exit()

if __name__ == '__main__':
    in_file = 'company_small.csv'
    # in_file = 'company_names.csv'
    in_file = "starbucks_test.csv"
    in_file = "chennai.csv"

    data, vocabulary = read_data(in_file)
    z = initialize_z(data)

    #print('Read {} companies...'.format(len(data)))
    prob_core, prob_back = load_probabilities()

    # if core.pickle and back.picles files already exist, do not retrain
    # prob_core = None
    # prob_back = None

    """print (vocabulary)
    print (vocabulary["gpo"])
    exit()"""

    if not prob_core or not prob_back:

        prob_core, prob_back = init_probabilities(vocabulary)

        print('Initializing EM...')
        prob_core, prob_back = expectation_maximization(data, vocabulary, prob_core, prob_back, z, max_iters=20)
        write(prob_core, prob_back)

    with open('probs.csv', 'w') as fout:
        fout.write('word, core, background\n')
        for w in vocabulary.keys():
            prob_core[w]
            prob_back[w]
            fout.write('{:20}, {:.5}, {:.5}\n'.format(w, prob_core[w], prob_back[w]))

    run_test(data, prob_core, prob_back)
