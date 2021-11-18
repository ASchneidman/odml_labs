import numpy as np
import numpy.random as random
import hashlib
from scipy.spatial.distance import hamming


class Enrolled:
    def __init__(self, name, shuffling_key, y):
        self.name = name
        self.shuffling_key = shuffling_key
        self.y = y

def shuffle(feature_vector, shuffling_key):
    # feature_vector is binary vector
    part1 = []
    part2 = []

    j = 0
    for i in range(shuffling_key.shape[0]):
        # assumes block size of 1
        if shuffling_key[i]:
            part1.append(feature_vector[j : j + 1])
        else:
            part2.append(feature_vector[j : j + 1])
        j += 1

    part1.extend(part2)
    return np.array(part1)

def query(query_vector, enrollment):
    query_vector = query_vector.flatten()
    # binarize the query vector
    query_vector = query_vector > np.median(query_vector)

    lowest_e = None
    dist = np.inf
    for e in enrollment:
        pred = shuffle(query_vector, e.shuffling_key)
        computed_dist = hamming(e.y, pred)
        if computed_dist < dist:
            dist = computed_dist
            lowest_e = e

    return lowest_e

def enroll(name, pin, feature_vector, enrollment):
    h = hashlib.sha256(pin.encode('utf-8'))
    m = int(h.hexdigest(), 16) % (10 ** 8)
    # Set the numpy random seed with the user's hash
    random.seed(m)

    shuffling_key = random.uniform(low=0.0, high=1.0, size=(feature_vector.shape[0])) < 0.5

    feature_vector = feature_vector.flatten()
    feature_vector = feature_vector > np.median(feature_vector)
    y = shuffle(feature_vector, shuffling_key)

    enrollment.append(Enrolled(name, shuffling_key, y))
    
        
"""
name1 = 'Tom'
pin1 = 'test1'
feature1 = random.uniform(low=0.0, high=1.0, size=(256,512))

name2 = 'John'
pin2 = 'test2'
feature2 = random.uniform(low=0.0, high=1.0, size=(256,512))

enrollment = []
enroll(name1, pin1, feature1, enrollment)
enroll(name2, pin2, feature2, enrollment)

assert(len(enrollment) == 2)

e = query(feature2, enrollment)
print(e.name)
"""