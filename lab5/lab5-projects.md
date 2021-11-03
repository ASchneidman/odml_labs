Lab 5: Group work on projects
===
The goal of this lab is for you to make progess on your project, together as a group. You'll set goals and work towards them, and report what you got done, chaellenges you faced, and subsequent plans.

Group name:
---
Group members present in lab today:

Alex, Navya, Bassam

1: Plan
----
1. What is your plan for today, and this week? 

Based on our results from the last lab (baselines), and feedback from the proposal, we've decided to pivot our project to focus on analyzing the impact of privacy-preserving techniques on Rezemblyzer, a diarization package we were able to run on device. 

3. How will each group member contribute towards this plan?

We plan on having each group member examine a different privacy-preserving technique. These will include cancelable biometrics (using per-user auxiliary data to generate transformations which can be revoked in the case of a leak) and cryptographic approaches (for example using a single hashing function on both the template and the queries), among others.

2: Execution
----
1. What have you achieved today / this week? Was this more than you had planned to get done? If so, what do you think worked well?

This week we completed benchmarks for on device speaker diarization which included experimenting with 6 different backends for generating speech embeddings. Getting the pipeline to run worked reasonably well, although our diarization error rate leaves much to be desired. We worked on improving the diarization error rate and have debugged a number of key memory issues to get our pipeline up and running. 

We've decided to pivot to experimenting with privacy-preserving techniques applicable to on device speaker diarization. We want to explore this route since it will provide a more novel angle to the project rather than just looking at performance tradeoffs between running these models on device. The novel aspect here will be looking at the performance impacts brought on by these different privacy-preserving approaches, since they will require additional memory for storing transformations or other hashes. 



3. Was there anything you had hoped to achieve, but did not? What happened? How did you work to resolve these challenges?

We hoped to see some better diarization performance from our baselines. Many of the backend models were not finetuned for diarization which likely caused some lack of quality in their embeddings. We saw some better performance by tweaking our clustering algorithms to better align the audio to the embeddings. 


4. What were the contributions of each group member towards all of the above?



3: Next steps
----
1. Are you making sufficient progress towards completing your final project? Explain why or why not. If not, please report how you plan to change the scope and/or focus of your project accordingly.

As mentioned above, we plan on pivoting to exploring privacy-preserving techniques in the diarization domain. This will add some novelty to the project with clear and easily comparable performance data. The overall goal of these approaches will be to protect the integrity of the system in the case of the user voice embedding templates (which are saved to disk) becoming leaked. For example, if someone is able to copy one of the stored templates, they would not be able to use that template to fool the system. 

3. Based on your work today / this week, and your answer to (1), what are your group's planned next steps?

We plan on exploring three different techniques. Each of these three will be divided up between each group member. We also plan on continuing to explore the literature to find other applicable techniques. 

We have three proposed techniques:

a. Cryptographic methods. This will entail using some hashing function to hash the query and template embeddings, and only store the hashed versions of the template embeddings.

b. User-Specific Random Projections. This will entail using a different random projection matrix for each user (with some additional properties) on the query and template embeddings. In the event a template becomes compromised, a new random projection matrix can be generated. See Teoh et al.

c. Sorted Index Number in combination with random projections. This approach creates a non-invertible transformation of feature vectors to vectors of indices which can then be used to compare against a dataset of templates. See Sorted Index Numbers for Privacy Preserving Face Recognition (Wang et al.). 

5. How will each group member contribute towards those steps? 

Navya will explore cryptographic methods, Bassam will explore user specific random projections, and Alex will explore Sorted Index Numbers in combination with random projections. 
