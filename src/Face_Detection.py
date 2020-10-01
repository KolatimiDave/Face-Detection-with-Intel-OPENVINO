import cv2
import logging as log
import numpy as np
import time
import os
from openvino.inference_engine import IENetwork, IECore


class Face_Detection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_xml_name, device='CPU', extensions=None, prob_thresh = 0.5):
        '''
        Use this to set your instance variables.
        '''
        self.plugin = None
        self.network = None
        self.device = device
        self.extensions = extensions
        self.exec_net = None
        self.input_blob = None
        self.input_shape = None
        self.output_blob = None
        self.output_shape = None
        self.model_xml = model_xml_name
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"
        self.prob_thresh = prob_thresh
            
    def check_model(self):
        '''
        Reads in the IR format of the model
        '''
        try:
            self.network = self.plugin.read_network(self.model_xml,self.model_bin)
        except Exception:
            raise ValueError('Error on Reading the IR, Ensure correct path to IR is given')
            
    def check_layers(self):
        '''
        Check for supported and unsupported layers in our network 
        '''
        try:
            supported_layers = self.plugin.query_network(network = self.network, device_name = self.device)
            unsupported_layers = [ i for i in self.network.layers.keys() if i not in supported_layers]
            if len(unsupported_layers)!=0 and self.device == 'CPU':
                self.plugin.add_extentions(self.extensions, self.device)
        except Exception:
            log.error('Unable to load the following unsupported layers')
            exit()

    def load_model(self): 
        self.plugin = IECore()   # Initialize the plugin
        self.check_model()       # load the IR
        self.check_layers()      # check supported and unsupported layers
        self.exec_net = self.plugin.load_network(network = self.network, device_name = self.device, num_requests = 1)    # Load the network into the plugin 
        self.input_blob = next(iter(self.network.inputs))      # Get the input layer
        self.input_shape = self.network.inputs[self.input_blob].shape       # Get the input shape
        self.output_blob = next(iter(self.network.outputs))    # Get the output layer
        self.output_shape = self.network.outputs[self.output_blob].shape    # Get the output shape

    def preprocess_input(self, image):
        '''
        preprocesses inputs before feeding into the model
        '''
        try:
            self.image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
            self.image = self.image.transpose((2,0,1))
            self.image = self.image.reshape(1, *self.image.shape)
            
        except Exception :
            log.error('Error occured while preprocessing the inputs | Face-Detection-Model')
            
        return self.image

    def preprocess_output(self, outputs, image):
        '''
        processeses the output of the predictions given by the model 
        '''
        points = []
        for box in outputs[0][0]:
            confidence = box[2]
            if confidence > self.prob_thresh:
                xmin = int(box[3] * image.shape[1])
                ymin = int(box[4] * image.shape[0])
                xmax = int(box[5] * image.shape[1])
                ymax = int(box[6] * image.shape[0])
                points.append((xmin,ymin,xmax,ymax))
                image = cv2.rectangle(image, (xmin, ymin),(xmax, ymax), (10,20,200), 1)
        return points, image

    def predict(self, image):
        '''
        performs the predictions given to the model
        '''
        try:
            preprocessed_image = self.preprocess_input(image)
            
            inference_time = time.time()
            results = self.exec_net.infer({self.input_blob:preprocessed_image})
            outputs = results[self.output_blob]
            inference_time = time.time() - inference_time
        
            self.points, self.image = self.preprocess_output(outputs,image)
            
            return image, inference_time

        except IndentationError:
            log.error('No Face has been detected | Face-Detection Model ')
            exit()