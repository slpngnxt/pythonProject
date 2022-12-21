import cv2
import time
from datetime import datetime
from multiprocessing import Process, Manager
from darknet.darknet import *
import shutil
import paho.mqtt.client as mqtt
import json
import pymysql

# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network("C:/Users/401-24/PycharmProjects/pythonProject/darknet/cfg/yolov4-csp.cfg", "C:/Users/401-24/PycharmProjects/pythonProject/darknet/cfg/coco.data", "C:/Users/401-24/PycharmProjects/pythonProject/darknet/yolov4-csp.weights")
width = network_width(network)
height = network_height(network)

conn = pymysql.connect(host='34.64.233.244', port=3306, user='root', passwd='qwer123', db='project',
                       charset='utf8',
                       autocommit=True)
def on_connect(client, userdata, flags, rc):
  if rc == 0:
    print("connected OK")
  else:
    print("Bad connection Returned code=", rc)
def on_disconnect(client, userdata, flags, rc=0):
  print("disconnect : ", str(rc))


def on_subscribe(client, userdata, mid, granted_qos):
  print("subscribed: " + str(mid) + " " + str(granted_qos))


def on_message(client, userdata, msg):
  print("got message :", str(msg.payload.decode("utf-8")))
  d = json.loads(msg.payload)
  cur = conn.cursor()
  cur.execute("""INSERT INTO video (user_id, video_path, file_name) 
    VALUES(%s, %s, %s)""", (d['user_id'], d['video_path'], '{}.mp4'.format(d['file_name'])))
  conn.commit()
  if (cur.rowcount):  # query 문의 성공여부를 확인, 0이면 변화 무
    print('데이터가 전송되었습니다. ')


def on_publish(client, userdata, mid):
  print("publish message :")

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
  d['video_path'] = '/static/'
  d['user_id'] = 'test9999'
  while d['r_finished'] == 0:
    if d['recording'] == 1:  # 레코딩중일 때는 계속 사진 찍기
      title = datetime.now().strftime('20%y-%m-%d_%H-%M-%S-%f')
      cv2.imwrite('C:/Users/401-24/PycharmProjects/pythonProject/darknet/content/{}/{}.jpg'.format(d['detected_time'], title), d['img_color'])

def init_mqtt():
  client = mqtt.Client()
  client.username_pw_set(username="admin", password="qwer123")

  client.on_connect = on_connect
  client.on_disconnect = on_disconnect
  client.on_subscribe = on_subscribe
  client.on_message = on_message
  client.on_publish = on_publish

  client.connect("34.64.233.244", 19883)
  client.subscribe('house/door', 1)

  return client
def bounding_box(d, l):
  import pymysql
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
  # print("클라이언트로 정보 전송")
  # client.loop_start()
  # client.publish('house/door',
  #                     '{"serial_number":"qwer123", "topic":"camera", "status":"person_detected", "Datetime":"22-12-16_12-12-12"}',
  #                     1)
  # client.loop_stop()
  # client.disconnect()
  client = init_mqtt()
  while d['r_finished'] == 0:
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

    if num_of_people > 0 and detected_person == 0:
      print("(1)새로운 사람이 등장하였습니다({}인)".format(num_of_people))
      detected_time = datetime.now()
      d['detected_time'] = detected_time.strftime('20%y-%m-%d_%H-%M-%S-%f')
      l.append(d['detected_time'])
      os.mkdir('C:/Users/401-24/PycharmProjects/pythonProject/darknet/content/{}'.format(d['detected_time']))
      d['recording'] = 1
      detected_person = 1

      # serial_number = 'qwer123'
      # topic = 'hy_py_camera'
      # status = 'person_detected'
      # Datetime = d['detected_time']
      json_str = '{"sensor":"door_sensor", "status":"door_open", "Datetime": null}'
      print("사람 탐지 -> 정보 전송")
      #client.loop_start()
      #client.loop_stop()
      json_object = json.loads(json_str)

      json_object['Datetime'] = d['detected_time']
      json_str = json.dumps(json_object)
      client.publish('house/door', json_str, 1)

      with conn.cursor() as cur:
        cur.execute("""INSERT INTO Video(user_id, video_path, file_name) 
                  VALUES(%s, %s, %s)""", (d['user_id'], d['video_path'], '{}.mpv'.format(d['detected_time'])))
        cur.execute("""INSERT INTO History(user_id, sensor, status, datetime) 
                  VALUES(%s, %s, %s, %s)""", (d['user_id'], "door_sensor", "door_open", detected_time))
        conn.commit()
        if (cur.rowcount):
          print("새로운 정보가 등록되었습니다.")

    elif num_of_people > 0 and detected_person == 1 and d['recording'] == 1:
      print("(2)사람이 감지되었지만 새로운 물체로 인식하지 않습니다(사람의 수는 변경되었을 수 있습니다)({}인)".format(num_of_people))

    else:
      print("(3)사람이 감지되지 않았습니다")
      detected_person = 0
      if d['recording'] == 1: # 사람이 감지되지 않았는데 녹화중이면 녹화를 중지하고 비디오를 만들어야 됨
        d['recording'] = 3 # 녹화 중지
        print("현재 감지된 시각 :", l)
        d['detected_total'] = d['detected_total'] + 1

    num_of_people = 0
# darknet helper function to run detection on image

def generate_video(d, l):
  import os
  import requests
  def send_file(filename):
    files = open('C:/Users/401-24/PycharmProjects/pythonProject/darknet/content/video/{}'.format(filename), 'rb')
    #files = open('C:/Users/401-24/PycharmProjects/pythonProject/darknet/content/video/testdog.jpg', 'rb')
    #files = open('C:/Users/401-24/PycharmProjects/pythonProject/darknet/content/video/video2022-12-21_09-52-14-365448.mp4', 'rb')
    upload = {'file': files}
    res = requests.post('http://34.64.233.244:9898/upload', files=upload)

  d['video_total'] = 0
  while True:
    if d['video_total'] < d['detected_total']:
      print('generate_video()가 실행됩니다')
      # d['making'] = 1
      image_folder = 'C:/Users/401-24/PycharmProjects/pythonProject/darknet/content/{}'.format(l[d['video_total']])
      print("확인...", l[int(d['video_total'])],
            print(os.path.isdir('C:/Users/401-24/PycharmProjects/pythonProject/darknet/content')))
      images = [img for img in os.listdir(image_folder)]
      images.sort()
      print(images)
      video_name = 'C:/Users/401-24/PycharmProjects/pythonProject/darknet/content/video/{}.mp4'.format(
        l[int(d['video_total'])])
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
      send_file('{}.mp4'.format(l[d['video_total']]))
      d['video_total'] = d['video_total'] + 1


    # if cv2.waitKey(1) & 0xFF == 27:
    #  break
    if d['r_finished'] == 1 and d['video_total'] == d['detected_total']:
      d['v_finished'] = 1
      break

if __name__ == '__main__':
  from darknet.darknet import *
  import os
  import pymysql

  with Manager() as manager:
    d = manager.dict()
    l = manager.list()

    p = Process(target=new_record, args=(d, l))
    p2 = Process(target=bounding_box, args=(d, l))
    p3 = Process(target=generate_video, args=(d, l))

    """
    con = pymysql.connect(host='34.64.39.224', port=3306, user='root', passwd='123', db='detection_history', charset='utf8',
                          autocommit=True)
    try:
      with con.cursor() as cur:
        cur.execute('select * from video_detections')
        rows = cur.fetchall()
        desc = cur.description
        print(f'{desc[0][0]:<30}{desc[1][0]:<10} {desc[2][0]:>20}')
        for row in rows:
          print(f'{row[0]:<30}{row[1]:<10} {row[2]:>20}')
    finally:
      con.close()
    """
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

    fourcc = cv2.VideoWriter_fourcc(*'X264')
    d['fourcc'] = fourcc

    d['ret'], d['img_color'] = cap.read()
    d['detected_total'] = 0
    d['r_finished'] = 0
    d['v_finished'] = 0

    # 모든 멀티프로세스를 실행시킴.
    # 각각의 프로세스는 while 무한 루프 + if / break문을 가지고 있기 때문에
    # 공유 메모리에 담기는 값에 따라 알아서 실행, 중지 될 것임
    p.start()
    p2.start()
    p3.start()

    while True:
      # 웹캠에서 이미지 읽어옴
      # 읽은 이미지는 frame 단위로 공유 메모리로 넘어감
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
        if d['detected_total'] < len(l):
          print("종료되지 않은 영상이 있습니다.")
          d['detected_total'] = d['detected_total'] + 1
        print("영상 감지를 종료합니다")
        d['r_finished'] = 1
        p.join()  # record
        print("recording이 join되었습니다")
        p2.join()  # bbox
        print("bbox가 join되었습니다")
        break


    while True:
      print("영상을 만드는 중입니다...")
      print("총 입퇴장 :", d['detected_total'], "제작된 비디오 수 : ", d['video_total'], "총 탐지 : ", len(l))
      time.sleep(0.5)
      if d['v_finished'] == 1:
        p3.join() # generate video
        print("generate video가 join되었습니다")
        cap.release()
        cv2.destroyAllWindows
        break

    conn.close()
