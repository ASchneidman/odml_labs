# https://www.merl.com/publications/docs/TR2011-077.pdf

import numpy as np
import numpy.random as random
import hashlib
from scipy.spatial.distance import hamming


class Enrolled:
    def __init__(self, name, A, w, template):
        self.name = name
        self.A = A
        self.w = w
        self.template = template


def query(query_vector, enrollment):
    query_vector = query_vector.flatten()

    # binarize the query vector
    query_vector = np.ceil(enrollment[0].A @ query_vector + enrollment[0].w) % 2

    lowest_e = None
    dist = np.inf
    for e in enrollment:
        computed_dist = hamming(e.template, query_vector)
        if computed_dist < dist:
            dist = computed_dist
            lowest_e = e

    return lowest_e

def enroll(name, pin, feature_vector, enrollment):
    feature_vector = feature_vector.flatten()
    if (len(enrollment) > 0):
        A, w = enrollment[0].A, enrollment[0].w
    else:
        A = np.diag(np.random.randn(512, feature_vector.shape[0]))
        w = np.random.uniform(high=1., size=(512,))

    template = np.ceil(A @ feature_vector + w) % 2


    enrollment.append(Enrolled(name, A, w, template))
    
        
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