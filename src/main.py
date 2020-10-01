from argparse import ArgumentParser
import os
import numpy as np
import logging as log
import time
import cv2
from Face_Detection import Face_Detection
from Input_feeder import InputFeeder


def main(args):
    video_input = args.video_input
    video_type = None
    feed = None

    if video_input != 'cam':
        if not os.path.isfile(video_input):
            log.error('path to video file does not exist')
            exit(1)
        video_type = 'video'
        feed = InputFeeder(input_type = video_type, input_file = video_input)
    elif video_input == 'cam':
        video_type = 'cam'
        feed = InputFeeder(input_type = video_type, input_file = video_input)
    else:
        log.error('Please enter either path to a video file or cam for web camera')


    model_load_time_start = time.time()
    model = Face_Detection(args.model, args.device, args.cpu_extension, args.prob_threshold)
    model.load_model()
    model_load_time = time.time() - model_load_time_start

    frame_count = 0
    feed.load_data()

    for flag, frame in feed.next_batch():
        if frame is None:
            log.error('could not read video input')
            exit()
        if not flag:
            break
        frame_count += 1

        frame, inference_time = model.predict(frame) # Model Prediction on Frame

        inf_time_message = 'Inference time: {:.3f}ms'\
            .format(inference_time)
        cv2.putText(frame, inf_time_message, (30,45), cv2.FONT_HERSHEY_COMPLEX, 1, (90, 180, 300), 3)

        if frame is not None:
            cv2.imshow('frame', frame)

        

    cv2.destroyAllWindows()
    feed.release()
    print('Face_Detection_model has a load time of {:.3f} seconds'.format(model_load_time))


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-v', '--video_input', required = True, type = str, help = 'enter path to video file or cam to use webcam')
    parser.add_argument('-m', '--model', required = True, type = str, help = 'Path to .xml file of pretrained face detection model')
    parser.add_argument('-d', '--device', type = str, default = 'CPU', required = False, help = 'specify target device to infer on, device can be: CPU, GPU, FPGA or MYRIAD. Default is CPU')
    parser.add_argument('-l', '--cpu_extension', type = str, default = None, required = False, help = 'specify path to cpu extension')
    parser.add_argument('-t', '--prob_threshold', required = False, type = float, default = 0.7, help = 'specify probaility threshold for the face detection model')


args = parser.parse_args()
main(args)