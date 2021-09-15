import cv2, os, sys
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

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



img = Image.open("alex.png")

mtcnn = MTCNN(select_largest=False, device='cuda')
face = mtcnn(img)

if face is not None:
    print(f"Detected a face! Its embedding is {face}")
