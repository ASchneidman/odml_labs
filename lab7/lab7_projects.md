Lab 7: Group work on projects
===
The goal of this lab is for you to make progess on your project, together as a group. You'll set goals and work towards them, and report what you got done, chaellenges you faced, and subsequent plans.

Group name:
---
Group members present in lab today:
Alex, Navya, Bassam

1: Plan
----
1. What is your plan for today, and this week? 

Today we met with the course staff to discuss our pivot to privacy preserving techniques. This week we plan on continuing to implement some more privacy preserving methods and pin down how we will benchmark these different techniques.

We decided on the following metrics to log and evaluate: Diarization Error Rate/Accuracy, power consumption, inference/query times for speaker diarization with and without biometric salting, and how well the secure version matches the insecure version in terms of predictions. We'll be varying parameters such as audio sampling frequency and hashing precision and logging all of the above metrics to evaluate performance and times.


3. How will each group member contribute towards this plan?

Last week, we implemented the biometric salting. This week, Alex finalized the speaker diarization demo. Bassam modified this demo so that it measures diarization accuracies and query times and stores them for plotting. He also wrote a script to take care of plotting. Navya took care of running the demo code on device and compiling the results for use in our final report later.

2: Execution
----
1. What have you achieved today / this week? Was this more than you had planned to get done? If so, what do you think worked well?

This week we were able to accomplish everything that we had planned to do. We have also decided to drop the power consumption metric, at least for now, because we don't have 

3. Was there anything you had hoped to achieve, but did not? What happened? How did you work to resolve these challenges?

No. There was a slight issue moving some files to the device but this was due to folder permissions being set to read-only. This has since been resolved.

5. What were the contributions of each group member towards all of the above?

All group members met with the course staff to discuss the above. 

3: Next steps
----
1. Are you making sufficient progress towards completing your final project? Explain why or why not. If not, please report how you plan to change the scope and/or focus of your project accordingly.

We are making sufficient progress towards completing the project. We have already implemented random projections, performance comparison techniques, and are working to debug any issues we currently have with logging and plotting these metrics. 

3. Based on your work today / this week, and your answer to (1), what are your group's planned next steps?

We need to fix some bugs in our logging code and then just start accumulating metrics for presentation in our final report.

5. How will each group member contribute towards those steps? 

Since we've already implemented all of the logging code. We just need to focus on . We'll each take turns running our logging demo to collected metrics and various sampling rates. Alex wants to also implement binary shuffling as an additional crytpographic method to compare.
