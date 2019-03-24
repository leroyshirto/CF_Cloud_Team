#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2
import time
import logging as log
from openvino.inference_engine import IENetwork, IEPlugin


class InferenceEngine(object):

    def __init__(self, model_bin, model_xml, device, async=False):

        log.basicConfig(format="[ %(levelname)s ] %(message)s",
                        level=log.INFO, stream=sys.stdout)
        # Plugin initialization for specified device and load extensions library if specified
        log.info("Initializing plugin for {} device...".format(device))
        cpu_extension = "/usr/local/lib/libcpu_extension.so"
        self.plugin = IEPlugin(device=device)
        if cpu_extension and 'CPU' in device:
            self.plugin.add_cpu_extension(cpu_extension)

        # Read IR
        log.info("Reading IR...")

        net = IENetwork(model=model_xml, weights=model_bin)

        if self.plugin.device == "CPU":
            supported_layers = self.plugin.get_supported_layers(net)
            not_supported_layers = [
                l for l in layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                          format(self.plugin.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in demo's command line parameters using -l "
                          "or --cpu_extension command line argument")
                sys.exit(1)
        assert len(net.inputs.keys(
        )) == 1, "This application supports only single input topologies"
        assert len(
            net.outputs) == 1, "This application supports only single output topologies"
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        log.info("Loading IR to the plugin...")
        self.exec_net = self.plugin.load(network=net, num_requests=2)

        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        del net

        self.async = async

        self.cur_request_id = 0
        self.next_request_id = 1

    def submit_request(self, frame, wait=True):

        in_frame = cv2.resize(frame, (self.w, self.h))
        # Change data layout from HWC to CHW
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))

        if self.async:
            self.exec_net.start_async(request_id=self.next_request_id, inputs={
                                      self.input_blob: in_frame})

        else:
            self.exec_net.start_async(request_id=self.cur_request_id, inputs={
                                      self.input_blob: in_frame})

        if wait:
            return wait()

    def wait(self):
        res = (self.exec_net.requests[self.cur_request_id].wait(-1) == 0)
        if res:
            if self.async:
                self.cur_request_id, self.next_request_id = self.next_request_id, self.cur_request_id
        return res

    def fetch_result(self):
        return self.exec_net.requests[self.cur_request_id].outputs[self.out_blob]
