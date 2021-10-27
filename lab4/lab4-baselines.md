Lab 4: Baselines and Related Work
===
The goal of this lab is for you to survey work related to your project, decide on the most relevant baselines, and start to implement them.

Ideally, the outcome of this lab would be: (1) the related work section of your project report is written and (2) baselines have been benchmarked.

Group name:
---
Group members present in lab today: Alex, Navya, Bassam

1: Related Work
----
1. Choose at least 2 pieces of related work per group member. For each piece of related work, write a paragraph that includes:
    - Summary of the main contributions of that work.
    - How your proposed project builds upon or otherwise relates to that work.



Another potential solution to the speaker diarization problem or "who spoke when" was introduced in 2019 in a joint effort betweent Google and Colombia University in their paper, Fully Supervised Speaker Diarization. In the paper, the authors introduce unbounded interleaved-state recurrent neural networks (UIS-RNN). The RNN accepts speaker-discriminative embeddings (a.k.a. d-vectors) from input utterances and each individual
speaker is modeled by a parameter-sharing RNN.

This RNN is naturally integrated with a distance-dependent Chinese restaurant process (ddCRP) to accommodate an unknown number of speakers. Our system is fully supervised and is able to learn from examples where time-stamped speaker labels are annotated. We achieved a 7.6% diarization error rate on NIST SRE 2000 CALLHOME, which is better than the state-of-the-art method using spectral clustering. Moreover, our method decodes in an online fashion while most state-of-the-art systems rely on offline clustering.


Speaker Diarization with Session Level Speaker Embedding Refinement Using Graph Neural Networks (Wang et. al, 2020) proposes using GNNs to produce more seperable embeddings for each speaker segment. Using a pretrained backend model to produce initial embeddings, this work then learns a mapping from each embedding to a graph, on top of which a number of graph neural network layers are used to produce refined embeddings. Affinity propogation is used to cluster these embeddings and produce an affinity matrix, which can then be used to assign identity to each segment.  



2: Baselines
----
1. What are the baselines that you will be running for your approach? Please be specific: data, splits, models and model variants, any other relevant information.

We attempted to use wav2vec, wav2vec 2.0, modified_cpc, vq_wav2vec, wav2vec_xlsr, and Resemblyzer.

All of the models are pretrained. For testing, however, we are using the VOX Converse Dataset. It contains 10 wav files of snippets from various news broadcasts. In each file, there are at most 4 unique speakers.

We load each of the models and then pass a downsampled version of the wav files into the model for embedding. We downsample each traditional 44 KHz wav file into having approximately 1 KHz sample rate. Without the downsampling, embedding the models took well over an hour each time.

2. These baselines should be able to run on your device within a reasonable amount of time. If you haven't yet tried to run them, please include a back-of-the-envelope calculation of why you think they will fit in memory. If the baselines will not fit in memory, return to (1) and adjust accordingly.




3. How will you be evaluating your baselines?

We are evaluating our baselines using the diarization error rate. Since speaker diarization attempts to segment an input audio stream by speaker identity, the DER is a measure as the fraction of time that is not attributed correctly to a speaker or to non-speech.

In other words, given an input audio signal, our networks can create 256-element vector embeddings for each 10 ms stride. Once the audio is converted into a list of embeddings, we attempt to find 4 clusters in the embeddings using scikit-learns K-Means clustering algorithm. We assign each vector one of the cluster categories and then compute the diarization error rate. 



4. Implement and run the baselines. Document any challenges you run into here, and how you solve them or plan to solve them.

Major difficulties included downloading and setting up s3prl library on-device. Despite it's promise of being a unified front-end framework that provides access to several audio embedding networks, s3prl is very poorly optimized for non-Intel systems as we discovered in the previous lab. 


5. If you finish running and evaluating your baselines, did the results align with your hypotheses? Are there any changes or focusing that you can do for your project based on insights from these results?



3: Extra
----
More related work, more baselines, more implementation, more analysis. Work on your project.


FAQ
----
1. Our baseline is the SotA model from XXX 2021 which doesn't fit on device.

Yikes! I think you meant to say -- "We used some very minimal off the shelf components from torchvision/huggingface/... to ensure our basic pipeline can run"

2. We're evaluating our baseline on accuracy only

I think you meant to say -- "We plan to plot accuracy against XXXX to see how compute and performance trade-off. Specifically, we can shrink our baseline by doing YYYY"

