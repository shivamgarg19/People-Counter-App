"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
	"""
	Parse command line arguments.

	:return: command line arguments
	"""
	parser = ArgumentParser()
	parser.add_argument("-m", "--model", required=True, type=str,
						help="Path to an xml file with a trained model.")
	parser.add_argument("-i", "--input", required=True, type=str,
						help="Path to image or video file")
	parser.add_argument("-l", "--cpu_extension", required=False, type=str,
						default=None,
						help="MKLDNN (CPU)-targeted custom layers."
							 "Absolute path to a shared library with the"
							 "kernels impl.")
	parser.add_argument("-d", "--device", type=str, default="CPU",
						help="Specify the target device to infer on: "
							 "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
							 "will look for a suitable plugin for device "
							 "specified (CPU by default)")
	parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
						help="Probability threshold for detections filtering"
						"(0.5 by default)")
	return parser


def connect_mqtt():
	# Connect to the MQTT client #
	client = mqtt.Client()
	client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
	return client

	
def preprocessing(frame, height, width):

	p_frame = cv2.resize(frame, (width, height))
	p_frame = p_frame.transpose((2, 0, 1))
	p_frame = p_frame.reshape(1, 3, height, width)
	
	return p_frame
	
def handle_output(frame, output, threshold_prob, input_shape):
	output = output[0][0]
	current_count = 0
	
	for l in range(len(output)):
		if output[l][2] > threshold_prob:
			x_min = int(output[l][3]*input_shape[1])
			y_min = int(output[l][4]*input_shape[0])
			x_max = int(output[l][5]*input_shape[1])
			y_max = int(output[l][6]*input_shape[0])
			cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 1)
			current_count = current_count + 1
	
	return frame, current_count

def infer_on_stream(args, client):
	"""
	Initialize the inference network, stream video to network,
	and output stats and video.

	:param args: Command line arguments parsed by `build_argparser()`
	:param client: MQTT client
	:return: None
	"""
	# Initialise the class
	infer_network = Network()
	# Set Probability threshold for detections
	prob_threshold = args.prob_threshold

	current_request_id = 0
	start_time = 0
	last_time = 0
	total_count = 0
	last_count = 0
	count_increase = 0
	count_decrease = 0
	single_image_mode = False
	
	# Load the model through `infer_network` #
	infer_network = Network()
	n, c, h, w = infer_network.load_model(args.model, args.device)
	
	# Handle the input stream #
	if args.input == 'CAM':
		input_stream = 0
		
	elif args.input.endswith('.png') or args.input.endswith('.jpg'):
		single_image_mode = True
		input_stream = args.input
		
	else:
		input_stream = args.input
		
	cap = cv2.VideoCapture(input_stream)
	
	if input_stream:
		cap.open(args.input)

	# Loop until stream is over #
	while cap.isOpened():
	
		# Read from the video capture #
		flag, frame = cap.read()
		
		if not flag:
			break
		
		key_pressed = cv2.waitKey(60)

		# Pre-process the image as needed #
		image = preprocessing(frame, h, w)

		# Start asynchronous inference for specified request #
		infer_network.exec_net(current_request_id, image)
		inference_start_time = time.time()
		# Wait for the result #
		if infer_network.wait(current_request_id) == 0:
			# Get the results of the inference request #
			output = infer_network.get_output(current_request_id)
			inference_time = time.time() - inference_start_time
			cv2.putText(frame, "Inference time " + str(format(inference_time, '.3f')) , (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
			# Extract any desired stats from the results #
			frame, current_count = handle_output(frame, output, prob_threshold, frame.shape)

			# Calculate and send relevant information on #
			# current_count, total_count and duration to the MQTT server #
			# Topic "person": keys of "count" and "total" #
			# Topic "person/duration": key of "duration" #
			# Increase person count only when model detect it in consecutive 3 frame to avoid false count(Happens because of detection error by model).
			if current_count > last_count:
				count_increase = count_increase + 1
				count_decrease = 0
				if count_increase > 3:
					start_time = time.time()
					total_count = total_count + current_count - last_count
					client.publish("person", json.dumps({"total": total_count}))
					last_count = current_count
				
			elif current_count < last_count:
				count_decrease = count_decrease + 1
				count_increase = 0
				if count_decrease > 3:
					duration = int(time.time() - start_time)
					client.publish("person/duration", json.dumps({"duration": duration}))
					last_count = current_count
				
			else:
				count_increase = 0
				count_decrease = 0
				
			client.publish("person", json.dumps({"count": last_count}))
			
			
			if key_pressed == 27:
				break

		# Send the frame to the FFMPEG server #
		sys.stdout.buffer.write(frame)  
		sys.stdout.flush()


		# Write an output image if `single_image_mode` #
		if single_image_mode:
			cv2.imwrite('output_image.jpg', frame)
			print("Succes")
			
	cap.release()
	cv2.destroyAllWindows()
	client.disconnect()

def main():
	"""
	Load the network and parse the output.

	:return: None
	"""
	# Grab command line args
	args = build_argparser().parse_args()
	# Connect to the MQTT server
	client = connect_mqtt()
	# Perform inference on the input stream
	infer_on_stream(args, client)


if __name__ == '__main__':
	main()
