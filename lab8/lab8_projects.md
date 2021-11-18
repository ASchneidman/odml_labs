Lab 8: Group work on projects
===
The goal of this lab is for you to make progess on your project, together as a group. You'll set goals and work towards them, and report what you got done, chaellenges you faced, and subsequent plans.

Group name:
---
Group members present in lab today:
Alex, Navya, Bassam

1: Plan
----
1. What is your plan for today, and this week? 

We plan to implement an additional cryptographic method called binary shuffling.  _____

2. How will each group member contribute towards this plan?



2: Execution
----
1. What have you achieved today / this week? Was this more than you had planned to get done? If so, what do you think worked well?

This week we were able to accomplish everything that we had planned to do. We have also decided to drop the power consumption metric, at least for now, because we don't have 

2. Was there anything you had hoped to achieve, but did not? What happened? How did you work to resolve these challenges?

No. There was a slight issue moving some files to the device but this was due to folder permissions being set to read-only. This has since been resolved.

3. What were the contributions of each group member towards all of the above?

All group members met with the course staff to discuss the above. 

3: Next steps
----
1. Are you making sufficient progress towards completing your final project? Explain why or why not. If not, please report how you plan to change the scope and/or focus of your project accordingly.

We are making sufficient progress towards completing the project. Alex has finished implementing the binary shuffling cryptographic technique and integrated into our pipeline. It was written in the same manner as the biometric salting so all that is necessary to switch between the two in benchmarking is simply to change one line of code. This method works by associating each enrolled user with a binary key. After binarizing the input feature vectors, a given feature vector can be queried against a given enrollment by shuffling the bits of the feature vector according to the shuffling key.

2. Based on your work today / this week, and your answer to (1), what are your group's planned next steps?



3. How will each group member contribute towards those steps? 

Since we've already implemented all of the logging code. We just need to focus on . We'll each take turns running our logging demo to collected metrics and various sampling rates. Alex wants to also implement binary shuffling as an additional crytpographic method to compare.
