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

We are also looking at another direction to make the current models work efficiently for the possible scenarios 

a. on-device extensions to handle the speaker-overlapping problem.  
b. effective real-time decoding techniques for speaker segmentation problems. 
c. work on distillation and quantisation based approaches to improve the energy consumption or latency.  

3. How will each group member contribute towards this plan?

We plan on having each group member examine a different privacy-preserving technique. These will include cancelable biometrics (using per-user auxiliary data to generate transformations which can be revoked in the case of a leak) and cryptographic approaches (for example using a single hashing function on both the template and the queries), among others. See the final section for more detail.

We plan on doing a critical analysis of the current end-to-end diarization models to identify the pain-points in having the current models work on our device. 

2: Execution
----
1. What have you achieved today / this week? Was this more than you had planned to get done? If so, what do you think worked well?

This week we completed benchmarks for on device speaker diarization which included experimenting with 6 different backends for generating speech embeddings. Getting the pipeline to run worked reasonably well, although our diarization error rate leaves much to be desired. We worked on improving the diarization error rate and have debugged a number of key memory issues to get our pipeline up and running.
 
We've decided to pivot to experimenting with privacy-preserving techniques applicable to on device speaker diarization. We want to explore this route since it will provide a more novel angle to the project rather than just looking at performance tradeoffs between running these models on device. The novel aspect here will be looking at the performance impacts brought on by these different privacy-preserving approaches, since they will require additional memory for storing transformations or other hashes. 


Typical speaker diarization systems are based on extraction and clustering of speaker representations. The system first extracts speaker representations such as i-vectors  or  d-vectors and then the speaker representations of short segments are partitioned into speaker clusters. We encountered numerous problems with the clustering based approaches.  Most of the speaker-change models assume one speaker for each segment, which hinders the application of the method for speaker-overlapping speech.

We have done a literature survey on end-to-end diarization models, analysed the pitfalls of the current models, and brainstormed further improvements to work on going forward. 


3. Was there anything you had hoped to achieve, but did not? What happened? How did you work to resolve these challenges?

We hoped to see some better diarization performance from our baselines. Many of the backend models were not finetuned for diarization which likely caused some lack of quality in their embeddings. We saw some better performance by tweaking our clustering algorithms to better align the audio to the embeddings. 

We planned to play around with one of the end-to-end models and analyze the performance and latency for speaker-overlapping datasets. But we have decided to work on it after getting feedback from the instructors on both the directions that we are looking into (privacy preservation techniques vs enhancing end-to-end models for on-device solutions).
 
4. What were the contributions of each group member towards all of the above?

Alex explored possible project ideas for privacy preservation techniques and came up with a detailed plan on different techniques to experiment with(if we decide to pursue that direction going forward).
 
Navya and Bassam did the literature survey on speaker re-segmentation approaches and brainstormed on various modelling improvements for speaker-overlapping problems and latency improvisation techniques. 


3: Next steps
----
1. Are you making sufficient progress towards completing your final project? Explain why or why not. If not, please report how you plan to change the scope and/or focus of your project accordingly.

As mentioned above, we plan on pivoting to exploring privacy-preserving techniques in the diarization domain. This will add some novelty to the project with clear and easily comparable performance data. The overall goal of these approaches will be to protect the integrity of the system in the case of the user voice embedding templates (which are saved to disk) becoming leaked. For example, if someone is able to copy one of the stored templates, they would not be able to use that template to fool the system. 


3. Based on your work today / this week, and your answer to (1), what are your group's planned next steps?

We plan on exploring three different techniques. Each of these three will be divided up between each group member. We also plan on continuing to explore the literature to find other applicable techniques. 

We have three proposed techniques for Privacy Preservation:

1. Cryptographic methods. This will entail using some hashing function to hash the query and template embeddings, and only store the hashed versions of the template embeddings.

2. User-Specific Random Projections. This will entail using a different random projection matrix for each user (with some additional properties) on the query and template embeddings. In the event a template becomes compromised, a new random projection matrix can be generated. See Teoh et al.

3. Sorted Index Number in combination with random projections. This approach creates a non-invertible transformation of feature vectors to vectors of indices which can then be used to compare against a dataset of templates. See Sorted Index Numbers for Privacy Preserving Face Recognition (Wang et al.). 

We have three ideas that we plan to experiment with for further enhancing the existing models for on-device solutions. 

The approach proposed in the paper (End-to-end speaker segmentation for overlap-aware resegmentation) has shown promising results in improving the diarization error rate on various datasets. It solves end-to-end speaker segmentation, instead of addressing voice activity detection, speaker change detection, and overlapped speech detection as three different tasks. Given an observation sequence X = (xt ∈ RF| t = 1, · · · , T) from an audio signal, it estimates the speaker label sequence Y = (yt | t = 1, · · · , T). P(Y |X) can be factored using the conditional independence assumption.

They formulate it as a multi-label classification problem and work under a threshold constraint on the maximum number of speakers that an audio can have. To cope with the label ambiguity problem, they introduce two permutation-free loss functions. First loss function is the permutation-invariant training (PIT) loss function, which is used for considering all the permutations of ground-truth speaker labels. The second loss function is the Deep Clustering (DPCL) loss function, which is used for encouraging hidden activations of the network to be speaker discriminative representations. 

Based on this architecture, we are planning to pursue the following ideas for our project. 

Experiment with underlying architecture to replace LSTMs with transformer based approaches to reduce the latency by getting rid of sequential architectures.  
The current SOTA models on end-to-end diarization are massive and we plan to work on knowledge distillation and compression methods to bring down the size of the models to make them suitable for on-device solutions.
Experiment with parallelizable online decoding techniques for speaker assignment problem  for faster computation of speaker embeddings (xt) to decode the speaker label at time ‘t’. 
 

5. How will each group member contribute towards those steps? 

Navya will explore cryptographic methods, Bassam will explore user specific random projections, and Alex will explore Sorted Index Numbers in combination with random projections.

If we decide to move with performance enhancement for on-device models, we are looking at the following plan of work. 

Navya will work on generating a dataset with diarization-style mixture simulation with varying speaker overlap rates: each speech mixture should have dozens of utterances per speaker with reasonable silence intervals between utterances. 

Alex will look at distillation and compression techniques for the existing models like (PyAnnote). 

Bassam will explore architecture specific tricks to improvise on latency. 


