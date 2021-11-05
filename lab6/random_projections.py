"""
@author: Bassam Bikdash


"""

import numpy as np
import numpy.random as random
import hashlib
from scipy.spatial.distance import hamming

class Enrolled():
    def __init__(self, name, pin_hash, pin_hash_i, R, y):
        self.name = name
        self.pin_hash = pin_hash
        self.pin_hash_i = pin_hash_i
        self.R = R
        self.y = y

def query(query_vector, enrollment):
    
    dist = np.inf
    lowest_e = enrollment[0]
    # Examine every enrolled person
    for e in enrollment:

        # Project the query vector into the space of the current enrolled person
        pred = e.R.T @ query_vector
        pred = np.where(pred<0, 0, 1).flatten()

        # Compare the distance of the current person audio to the projected query vector
        computed_dist = hamming(e.y, pred)
        if computed_dist < dist
            # Update the lowest dist
            dist = computed_dist
            lowest_e = e

    # Return the enrollment object of the person whose closest to the query
    return lowest_e


def enroll(pin, feature_vector, enrollment):

    # Create a hash of the user's pin and convert it to an integer
    h = hashlib.sha256(pin.encode('utf-8'))
    m = int(h.hexdigest(), 16) % (10 ** 8)
    # Set the numpy random seed with the user's hash
    random.seed(m)

    # Create a random
    X = random.uniform(low=0.0, high=1.0, size=(feature_vector.shape[0],512))

    # Perform gram-schmidt orthonormalization on X
    Q, R = np.linalg.qr(X)

    y = Q.T @ feature_vector
    y = np.where(y<0, 0, 1).flatten()

    enrollment.append(Enrolled('test', h, m, Q, y))
    return


pin = 'test'
feature = random.uniform(low=0.0, high=1.0, size=(256,512))
enrollment = []
enroll(pin, feature, enrollment)
query()

"""
1) Each user inputs a pin/password and a feature vector, N. We compute a hash of the pin/password then use the hash as a seed to
to generate a random matrix R, NxN for each user. pin/password -> R
2) We apply R to each users feature vector to get the binary vector mentioned here: http://www.scholarpedia.org/article/Cancelable_biometrics
3) Then we store a file with each user's hash and R, transformed feature vector which will serve as enrollment

dict = { hash1: R1, binary_vector1
         hash2: R2, binary_vector2,
         hash3: R3, binary_vector3}
"""
