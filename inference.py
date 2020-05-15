#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IECore


class Network:
	"""
	Load and configure inference plugins for the specified target devices 
	and performs synchronous and asynchronous modes for the specified infer requests.
	"""

	def __init__(self):
		# Initialize any class variables desired #
		self.plugin = None
		self.input_blob = None
		self.output_blob = None
		self.exec_network = None
		self.infer_request_handle = None

	def load_model(self, model, device="CPU"):
		# Load the model #
		model_xml = model
		model_bin = os.path.splitext(model)[0] + ".bin"
		
		self.plugin = IECore()
		
		# IENetwork Class is depreceted in Openvino2020 #
		network = self.plugin.read_network(model=model_xml, weights= model_bin)
		
		# Check for supported layers #
		supported_layers = self.plugin.query_network(network=network, device_name=device)
		unsupported_layers = [l for l in network.layers.keys() if l not in supported_layers]
		if len(unsupported_layers) != 0:
			print("Unsupported layers found: {}".format(unsupported_layers))
			print("Check whether extensions are available to add to IECore.")
			exit(1)

		self.input_blob = next(iter(network.inputs))
		self.out_blob = next(iter(network.outputs))
		
		# Add any necessary extensions #
		# Return the loaded inference plugin #
		self.exec_network = self.plugin.load_network(network, device)
		
		return self.get_input_shape(network)

	def get_input_shape(self, network):
		# Return the shape of the input layer #
		return network.inputs[self.input_blob].shape

	def exec_net(self, request_id, frame):
		# Start an asynchronous request #
		# Return any necessary information #
		self.infer_request_handle = self.exec_network.start_async(request_id=request_id, inputs={self.input_blob: frame})
		return self.exec_network

	def wait(self, request_id):
		# Wait for the request to be complete. #
		# Return any necessary information #
		return self.exec_network.requests[request_id].wait(-1)

	def get_output(self, request_id):
		# Extract and return the output results
		return self.exec_network.requests[request_id].outputs[self.out_blob]
