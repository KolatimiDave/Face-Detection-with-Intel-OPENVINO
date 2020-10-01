import cv2


class InputFeeder:
    def __init__(self, input_type, input_file=None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.input_type = input_type
        if input_type == 'video' or input_type == 'image':
            self.input_file = input_file

    def load_data(self):
        if self.input_type == 'video':
            self.cap=cv2.VideoCapture(self.input_file)
        elif self.input_type == 'cam':
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.imread(self.input_file)
        if not self.cap.isOpened():
            exit(0)

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        while True:
            flag, frame = self.cap.read()
            if not flag:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            yield flag, frame

    def calculate_fps(self):
        '''
        finds frames per second
        '''
        return int(self.cap.get(cv2.CAP_PROP_FPS))

    def release(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type=='image':
            self.cap.release()
        pass