# https://arxiv.org/pdf/1510.05937.pdf

import numpy as np
import numpy.random as random
import hashlib
from scipy.spatial.distance import hamming


class Enrolled:
    def __init__(self, name, projections, binary_template):
        self.name = name
        self.projections = projections
        self.binary_template = binary_template


def query(query_vector, enrollment):
    query_vector = query_vector.flatten()

    # All enrolled users have same projection matrix
    query_vector_binary = (query_vector @ enrollment[0].projections) >= 0

    lowest_e = None
    dist = np.inf
    for e in enrollment:
        computed_dist = hamming(e.binary_template, query_vector_binary)
        if computed_dist < dist:
            dist = computed_dist
            lowest_e = e

    return lowest_e

def enroll(name, pin, feature_vector, enrollment):
    feature_vector = feature_vector.flatten()
    # pin is not used here
    if len(enrollment) > 0:
        projections = enrollment[0].projections
    else:
        projections = np.random.randn(feature_vector.shape[0], feature_vector.shape[0])

    binary_template = (feature_vector @ projections) >= 0
    enrollment.append(Enrolled(name, projections, binary_template))
    
        
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