import socket
import cv2
import pickle
import argparse
import mmcv
from mmdet.apis import inference_detector, init_detector
import onnxruntime
import numpy as np
import struct


def parse_args():
    parser = argparse.ArgumentParser(description='Video demo')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='The interval of show (s), 0 is block')
    parser.add_argument(
        '--backend',
        choices=['cv2', 'plt'],
        default='cv2',
        help='the backend to visualize the results')
    parser.add_argument('--fps', help='FPS of the output video')
    args = parser.parse_args()
    return args


HOST = ''
PORT = 8485

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')
s.bind((HOST, PORT))
print('Socket bind complete')
s.listen(10)
print('Socket now listening')
args = parse_args()
config = '../configs/scrfd/scrfd_500m.py'
checkpoint = '../weights/model_500.pth'
model = init_detector(config, checkpoint, device=args.device)
head = onnxruntime.InferenceSession('head.onnx')
conn, addr = s.accept()

data = b""
payload_size = struct.calcsize(">L")
print("payload_size: {}".format(payload_size))
count = 0
while True:
    while len(data) < payload_size:
        data += conn.recv(4096)
    # receive image row data form client socket
    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_msg_size)[0]
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]
    # unpack image using pickle
    frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    # video_writer = None
    # if args.out:
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     video_writer = cv2.VideoWriter(
    #         args.out, fourcc, 30,
    #         (frame.shape[0], frame.shape[1]))
    result = inference_detector(model, frame)
    result = [i for i in result[0] if i[-1] > args.score_thr]

    for box in result:
        (startX, startY, endX, endY, conf) = box.astype("int")
        face = frame[startY:endY, startX:endX]
        face = cv2.resize(face, (224, 224))
        face = np.expand_dims(face, axis=0).astype(np.float32)
        predict = head.run(["dense_1", "ID_Classifier"], {"input": face})
        mask = np.argmax(predict[0][0])
        pred_id = np.argmax(predict[1][0])
        label = "Mask" if mask == 1 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        person = 'R10922A23' if pred_id == 0 else 'R10922A17' if pred_id == 1 else 'D10922034' if pred_id == 2 else pred_id
        label = "{}: {:.2f}% | ID: {}".format(label, predict[0][0][mask] * 100, person)

        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    if args.out:
        cv2.imwrite(f'./test/{count}.png', frame)
        count+=1

    cv2.imshow('server', frame)
    cv2.waitKey(1)
    #
    # if video_writer:
    #     video_writer.release()