import cv2
import argparse
from mmdet.apis import inference_detector, init_detector
import mmcv
import numpy as np
import onnxruntime


def parse_args():
    parser = argparse.ArgumentParser(description='Video demo')
    parser.add_argument('video', help='Video file')
    parser.add_argument('mode', help='run mode')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='Bbox score threshold')
    parser.add_argument('--out', type=str, help='Output video file')
    parser.add_argument('--show', action='store_true', help='Show video')
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


def detect():
    args = parse_args()
    assert args.out or args.show, \
        ('Please specify at least one operation (save/show the '
         'video) with the argument "--out" or "--show"')
    config = 'configs/scrfd/scrfd_500m.py'
    checkpoint = 'weights/model_500.pth'
    model = init_detector(config, checkpoint, device=args.device)
    head = onnxruntime.InferenceSession('weights/head.onnx')

    video_reader = mmcv.VideoReader(args.video)
    video_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.out, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))

    for frame in mmcv.track_iter_progress(video_reader):
        result = inference_detector(model, frame)
        result = [i for i in result[0] if i[-1] > args.score_thr]
        print(result)
        for box in result:
            (startX, startY, endX, endY, conf) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (224, 224))
            # face = preprocess_input(face)
            face = np.expand_dims(face, axis=0).astype(np.float32)
            predict = head.run(["dense_1", "ID_Classifier"], {"input": face})
            mask = np.argmax(predict[0][0])
            pred_id = np.argmax(predict[1][0])

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask == 1 else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            person = 'R10922A23' if pred_id == 0 else 'R10922A17' if pred_id == 1 else 'D10922034' if pred_id == 2 else pred_id
            label = "{}: {:.2f}% | ID: {}".format(label, predict[0][0][mask] * 100, person)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)

        if args.show:
            cv2.namedWindow('video', 0)
            mmcv.imshow(frame, 'video', args.wait_time)
        ##mmcv.imshow(face, 'video', args.wait_time)
        if args.out:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'detect':
        detect()
