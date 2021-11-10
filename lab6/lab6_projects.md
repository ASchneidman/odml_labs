Lab 5: Group work on projects
===
The goal of this lab is for you to make progess on your project, together as a group. You'll set goals and work towards them, and report what you got done, chaellenges you faced, and subsequent plans.

Group name:
---
Group members present in lab today: Alex, Bassam, Navya

1: Plan
----
1. What is your plan for today, and this week? 

We plan on starting some of the implementations of the privacy preserving techniques specified in the last lab.

3. How will each group member contribute towards this plan?

Bassam began implement one of the methods of cancelable biometrics for encrypting user data. Alex began working on real time diarization with Rezemblyzer. Navya worked on the implementation of some diarization extensions (multiple speakers in one segment). 

2: Execution
----
1. What have you achieved today / this week? Was this more than you had planned to get done? If so, what do you think worked well?

Bassam implemented the biometric salting described at this link: http://www.scholarpedia.org/article/Cancelable_biometrics. Biometric salting is similar to password salting in cryptography, where a set of random bits r are concatenated to a secret key, k, associated with each user. The output is often stored as hash H(r+k) in the database. Biometric salting follows the same principle such that a user-specific and independent input factor (auxiliary data such as a password or user-specific random numbers) is blended with biometric data to derive a distorted version of the biometric template. Since the auxiliary data is externally derived and interacts directly with biometric data, it can be changed and revoked easily but must be kept secret for maximum security protection. However, since the external confidential keys or passwords are easily to be lost, stolen or compromised, the accuracy and vulnerabilities of existing schemes should be justified.

We implemented a specific instance of biometrics salting based on user specific random projection. Basically, we generate a random projection matrix from some auxiliary data, for example a user pin or password. Gram-Schmidt orthonormalization is then carried out on the on the matrix such that the matrix columns are orthonormal.

We then project a feature vector, x, by premultiplying the random projection matrix R, to retrieve a projected feature vector, y. y is then thresholded converted to a binary vector such that b_i=0 if y_i< Tau, otherwise b_i=1.

Here is an explanation of how we use this biometric salting in our project. Speaker diarization consists of two parts: speaker enrollment and speaker verification. During speaker enrollment, we store a voice print (aka. a neural network embedding of a recording of a person’s voice) of a person and use it as a template to compare future feature embedding to at runtime. If we store a person’s voice or it’s embedding in a raw form, this is not very secure.

Thus, we apply biometric salting to project the feature vector into an encrypted space where it can’t be easily decoded into the original form.
To actually implement this we did the following:

1.  Input a user pin, user name, and feature vector for enrollment
2.  Convert the pin to a hash and use the hash as a random seed to generate a matrix of random values. We used numpy.random for our seeding and to generate our random matrix, R.
3.  Perform Gram-Schmidt orthonormalization on R and then premultiply it with the feature vector for enrollment.
4.  Threshold values in the vector to produce a binary vector and save that vector. The original feature vector is not stored.
5.  For speaker verification, we apply the same biometric salting process to a query vector. The query is mapped into the projection space of each user and we compute the hamming distance between the transformed query and the template of each user. Hamming distance is a metric used to detect the closeness of binary vector.
6.  We return the user name whose template has the lowest hamming distance to the query.

We verified that our functions both project a query vector into the right subspace and that our function returns the user with the lowest hamming distance.

2. Was there anything you had hoped to achieve, but did not? What happened? How did you work to resolve these challenges?

No. These were the main goals of this particular lab of the project. Now, we are focused on integrating this method of cryptography into our primary speaker diarization pipeline.

3. What were the contributions of each group member towards all of the above?

Bassam implemented the function necessary for biometric salting. Alex integrated those functions in the primary speaker diarization pipeline. Navya


3: Next steps
----
1. Are you making sufficient progress towards completing your final project? Explain why or why not. If not, please report how you plan to change the scope and/or focus of your project accordingly.


2. Based on your work today / this week, and your answer to (1), what are your group's planned next steps?


3. How will each group member contribute towards those steps? 
