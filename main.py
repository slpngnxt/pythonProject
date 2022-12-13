import cv2
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

def new_record(d, l):
  d['recording'] = 0
  d['making'] = 0
  while True:
    if d['recording'] == 1:  # 레코딩중일 때는 계속 사진 찍기
      title = datetime.now().strftime('%y-%m-%d_%H-%M-%S-%f')
      cv2.imwrite('C:/Users/401-24/PycharmProjects/pythonProject/darknet/content/{}/outputIn({}).jpg'.format(d['detected_time'], title), d['img_color'])
    if cv2.waitKey(1) & 0xFF == 27:
      break

def bounding_box(d, l):

  def darknet_helper(img, width, height):
    darknet_image = make_image(width, height, 3)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height),
                             interpolation=cv2.INTER_LINEAR)

    # get image ratios to convert bounding boxes to proper size
    img_height, img_width, _ = img.shape
    width_ratio = img_width / width
    height_ratio = img_height / height

    # run model on darknet style image to get detections
    copy_image_from_bytes(darknet_image, img_resized.tobytes())
    detections = detect_image(network, class_names, darknet_image)
    free_image(darknet_image)
    return detections, width_ratio, height_ratio

  detected_person = 0
  num_of_people = 0

  p3 = Process(target=generate_video, args=(d, l))
  while True:
    detections, width_ratio, height_ratio = darknet_helper(d['img_color'], width, height)
    for label, confidence, bbox in detections:
      left, top, right, bottom = bbox2points(bbox)
      left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(
        bottom * height_ratio)
      cv2.rectangle(d['img_color'], (left, top), (right, bottom), class_colors[label], 2)
      cv2.putText(d['img_color'], "{} [{:.2f}]".format(label, float(confidence)),
                  (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                  class_colors[label], 2)
      print("라벨 : ", label)

      # 사람이 인식 됐을 때
      if float(confidence) > 0.8 and label == 'person':
        print('바운딩박스 사람 인식!')
        num_of_people = num_of_people + 1
    cv2.imshow('fdsf', d['img_color'])
    if num_of_people > 0 and detected_person == 0:
      print("(1)새로운 사람이 등장하였습니다({}인)".format(num_of_people))
      d['detected_time'] = datetime.now().strftime('%y-%m-%d_%H-%M-%S-%f')
      l.append(d['detected_time'])
      os.mkdir('C:/Users/401-24/PycharmProjects/pythonProject/darknet/content/{}'.format(d['detected_time']))
      d['recording'] = 1
      detected_person = 1


    elif num_of_people > 0 and detected_person == 1 and d['recording'] == 1:
      print("(2)사람이 감지되었지만 새로운 물체로 인식하지 않습니다(사람의 수는 변경되었을 수 있습니다)({}인)".format(num_of_people))

    else:
      print("(3)사람이 감지되지 않았습니다")
      detected_person = 0
      if d['recording'] == 1: # 사람이 감지되지 않았는데 녹화중이면 녹화를 중지하고 비디오를 만들어야 됨
        d['recording'] = 3 # 녹화 중지
        print("현재 감지된 시각 :", l)
        d['detected_total'] = d['detected_total'] + 1
        """
        # 만약 generate_video()가 실행중이지 않다면
        while True:
          if d['making'] == 0:
            p3.start()
            p3.join()
            break
        """

    num_of_people = 0
    if cv2.waitKey(1) & 0xFF == 27:
      p3.join()
      break
# darknet helper function to run detection on image

def generate_video(d, l):
    d['video_total'] = 0
    import os
    while True:
      if d['video_total'] < d['detected_total']:
        print('generate_video()가 실행됩니다')
        # d['making'] = 1
        image_folder = 'C:/Users/401-24/PycharmProjects/pythonProject/darknet/content/{}'.format(l[d['video_total']])
        print("확인...", l[int(d['video_total'])], print(os.path.isdir('C:/Users/401-24/PycharmProjects/pythonProject/darknet/content')))
        images = [img for img in os.listdir(image_folder)]
        images.sort()
        print(images)
        video_name = 'C:/Users/401-24/PycharmProjects/pythonProject/darknet/content/video/video({}).mp4'.format(l[int(d['video_total'])])
        frame = cv2.imread(os.path.join(image_folder, images[0]))

        height, width, layers = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, d['fps'], (width, height))

        for image in images:  # 5장 -> 1초 : 30초 -> 150장 615/150 => 2분정도 되는 frame
          video.write(cv2.imread(os.path.join(image_folder, image)))
        video.release()
        d['recording'] = 0
        if os.path.exists('C:/Users/401-24/PycharmProjects/pythonProject/darknet/content/{}'.format(l[d['video_total']])):
          shutil.rmtree('C:/Users/401-24/PycharmProjects/pythonProject/darknet/content/{}'.format(l[d['video_total']]))
        d['video_total'] = d['video_total'] + 1

      if cv2.waitKey(1) & 0xFF == 27:
        break


if __name__ == '__main__':
  from darknet.darknet import *
  import os
  with Manager() as manager:
    d = manager.dict()
    l = manager.list()

    p = Process(target=new_record, args=(d, l))
    p2 = Process(target=bounding_box, args=(d, l))
    p3 = Process(target=generate_video, args=(d, l))

    # 카메라는 하나
    cap = cv2.VideoCapture(0)

    # 웹캠에서 fps 값 획득
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps', fps)
    d['fps'] = fps

    if fps == 0.0:
      fps = 30.0

    time_per_frame_video = 1 / fps
    d['time_per_frame_video'] = time_per_frame_video
    last_time = time.perf_counter()
    d['last_time'] = last_time

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    d['width'] = width
    d['height'] = height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    d['fourcc'] = fourcc

    p.start()
    p2.start()
    p3.start()

    d['ret'], d['img_color'] = cap.read()
    d['detected_total'] = 0

    while True:

      # 웹캠에서 이미지 읽어옴
      d['ret'], img_color = cap.read()
      d['img_color'] = img_color

      if d['ret'] == False:
        print('웹캠에서 영상을 읽을 수 없습니다.')
        break

      # fsp 계산
      time_per_frame = time.perf_counter() - last_time
      time_sleep_frame = max(0, time_per_frame_video - time_per_frame)
      time.sleep(time_sleep_frame)

      real_fps = 1 / (time.perf_counter() - last_time)
      last_time = time.perf_counter()

      x = 30
      y = 50
      text = '%.2f fps' % real_fps

      # 이미지의 (x, y)에 텍스트 출력
      img_color = d['img_color']
      draw_text(img_color, text, x, y)


      cv2.imshow("Person Detected!", img_color)

      # ESC키 누르면 중지
      if cv2.waitKey(1) & 0xFF == 27:
        break
    p.join()
    p2.join()
    p3.join()
    cap.release()
    cv2.destroyAllWindows

