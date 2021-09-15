Lab 0: Hardware setup
===
The goal of this lab is for you to become more familiar with the hardware platform you will be working with this semester, and for you to complete basic setup so that everyone in the group should be able to work remotely on the device going forward. By the end of class today, everyone in your group should be able to ssh in to the device, use the camera to take a picture, record audio, run a basic NLP model, and run a basic CV model. 

If you successfully complete all those tasks, then your final task is to write a script that pipes together I/O with a model. For example, you could write a script that uses the camera to capture an image, then runs classification on that image. Or you could capture audio, run speech-to-text, then run sentiment analysis on that text.

Group name:
---
Group members present in lab today: Alex, Bassam, Navya

1: Set up your device.
----
Depending on your hardware, follow the instructions provided in this directory: [Raspberry Pi 4](https://github.com/strubell/11-767/blob/main/labs/lab0-setup/setup-rpi4.md), [Jetson Nano](https://github.com/strubell/11-767/blob/main/labs/lab0-setup/setup-jetson.md), [Google Coral](https://coral.ai/docs/dev-board/get-started/). 
1. What device(s) are you setting up?

Jetson Nano 2gb

2. Did you run into any roadblocks following the instructions? What happened, and what did you do to fix the problem?

We had some issues flashing the correct image onto the SD card. These were resolved by reformatting the SD card and flashing the correct image. We had some other issues with the camera (like it only giving us green images), this was resolved by following other student's solutions on the slack.

3. Are all group members now able to ssh in to the device from their laptops? If not, why not? How will this be resolved?

Yes, all group members can ssh into the machine. 

2: Collaboration / hardware management plan
----
4. What is your group's hardware management plan? For example: Where will the device(s) be stored throughout the semester? What will happen if a device needs physical restart or debugging? What will happen in the case of COVID lockdown?

The device will be stored in Bassam's lab, on CMU's campus. In the case of physical restart, the closest available group member will visit the device. In the case of COVID lockdown, Bassam should still have access to his lab. We will retrieve the device and set it up in one of our apartments. 


3: Putting it all together
----
5. Now, you should be able to take a picture, record audio, run a basic computer vision model, and run a basic NLP model. Now, write a script that pipes I/O to models. For example, write a script that takes a picture then runs a detection model on that image, and/or write a script that runs speech-to-text on audio, then performs classification on the resulting text. Include the script at the end of your lab report.

6. Describe what the script you wrote does (document it.) 

`face_detect.py` takes two optional arguments: a path or `--capture`. If a path is provided, an image is loaded from the given path. If `--capture` is set, a new image is taken from the camera. In either case, if a face is detected, its embedding is compared against an embedding of Alex's face, and the euclidean distance is outputted. Also, copy of the image is outputted with landmarks annotated over the face. 

`generate_text.py` loads a pipeline for summarization from Huggingface Transfomers library. It loads Distill-BART model to generate the summary of a given text.

7. Did you have any trouble getting this running? If so, describe what difficulties you ran into, and how you tried to resolve them.

We had some issues with camera that solutions from the slack helped us resolve. We also had some issues with torchvision. This was resolved by installing torchvision from a pre-built wheel designed for the jetson. 
