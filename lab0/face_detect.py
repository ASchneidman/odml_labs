import cv2, os, sys
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import argparse
import torch
from matplotlib import pyplot as plt

def gstreamer_pipeline(capture_width=1280, capture_height=720, 
                       display_width=1280, display_height=720,
                       framerate=60, flip_method=0):
  return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

def capture_image():
  HEIGHT=1280
  WIDTH=1920
  center = (WIDTH / 2, HEIGHT / 2)
  M = cv2.getRotationMatrix2D(center, 180, 1.0)

  cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

  img = None
  if cam.isOpened():
    val, img = cam.read()
    if val:
      cv2.imwrite('output.png', img)

  return img

parser = argparse.ArgumentParser(description="Detects faces and produces embeddings for those faces. If neither argument is provided, Alex will be detected on alex.png")
parser.add_argument('--path', help="Path to an image to detect a face on and output an embedding for that face.", default=None)
parser.add_argument('--capture', action='store_true', default=False, help="Capture a new image and detect a face on that image. If provided, path will be ignored. The image will be outputted as output.png")
parser.add_argument('--cuda', action='store_true', default=False, help="Use cuda for face detection. May be a lot slower. Default is False.")

args = parser.parse_args()

if args.capture:
  img = capture_image()
elif args.path is not None:
  img = Image.open(args.path)
else:
  img = Image.open("alex.png")

if args.cuda:
  mtcnn = MTCNN(select_largest=False, device='cuda')
else:
  mtcnn = MTCNN(select_largest=False, device='cpu')

face, prob = mtcnn(img, return_prob=True)
boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

alex_emb = mtcnn(Image.open("alex.png"))


print(f"Face probability is {prob}")
if face is not None:
  dist = (face-alex_emb).norm()
  print("Detected a face!")
  print(f"The distance between this face and alex is {dist}")

  fig, ax = plt.subplots(figsize=(16, 12))
  ax.imshow(img)
  ax.axis('off')

  for box, landmark in zip(boxes, landmarks):
    ax.scatter(*np.meshgrid(box[[0, 2]], box[[1, 3]]))
    ax.scatter(landmark[:, 0], landmark[:, 1], s=8)
  plt.savefig("landmarks.png")
else:
  print("Did not detect a face. Try taking off your mask?")
