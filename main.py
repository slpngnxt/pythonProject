import time
from datetime import datetime
from multiprocessing import Process, Manager
from darknet.darknet import *
import shutil
import os

# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network("C:/Users/401-24/PycharmProjects/pythonProject/darknet/cfg/yolov4-csp.cfg", "C:/Users/401-24/PycharmProjects/pythonProject/darknet/cfg/coco.data", "C:/Users/401-24/PycharmProjects/pythonProject/darknet/yolov4-csp.weights")
width = network_width(network)
height = network_height(network)

def draw_text(img, text, x, y):
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 1
  font_thickness = 2
  text_color = (255, 0, 0)
  text_color_bg = (0, 0, 0)

  text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
  text_w, text_h = text_size
  offset = 5

  cv2.rectangle(img, (x - offset, y - offset), (x + text_w + offset, y + text_h + offset), text_color_bg, -1)
  cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
