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


Alex worked on 1) resetting up the Nvidia Jetson Nano with our new 128 GB SD card 2) :



Navya worked on 1) testing the baseline diarization accuracy for Resymblyzer and 2):



Bassam worked on 1) baseline diarization accuracy for  the benchmarking script that Navya wrote and 2) getting the baseline diarization error rates for the following models:



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

