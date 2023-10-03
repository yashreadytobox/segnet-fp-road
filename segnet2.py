#!/usr/bin/env python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse
import torch

from jetson_inference import segNet
from jetson_utils import videoSource, videoOutput, cudaOverlay, cudaDeviceSynchronize, Log

from segnet_utils import *
import time

load_time1 = time.time()
# parse the command line
parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=segNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--visualize", type=str, default="mask", help="Visualization options (can be 'overlay' 'mask' 'overlay,mask'")
parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--alpha", type=float, default=150.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 150.0)")
parser.add_argument("--stats", action="store_true", help="compute statistics about segmentation mask class output")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# load the segmentation network
net = segNet(args.network, sys.argv)

# note: to hard-code the paths to load a model, the following API can be used:
#
# net = segNet(model="model/fcn_resnet18.onnx", labels="model/labels.txt", colors="model/colors.txt",
#              input_blob="input_0", output_blob="output_0")

# set the alpha blending value
net.SetOverlayAlpha(args.alpha)

# create video output
output = videoOutput(args.output, argv=sys.argv)

# create buffer manager
buffers = segmentationBuffers(net, args)

# create video source
input = videoSource(args.input, argv=sys.argv)
load_time2 = time.time()
print(f"LOAD TIME = {load_time2 - load_time1} sec")
# process frames until EOS or the user exits
while True:
    t1 = time.time()
    # capture the next image
    img_input = input.Capture()
    tn1 = time.time()
    print(f'IMAGE CAPTURE TIME: {tn1-t1}')

    if img_input is None: # timeout
        continue
    tn2 = time.time()
    print(f'TIMEOUT: {tn2-tn1}')
    # allocate buffers for this size image
    buffers.Alloc(img_input.shape, img_input.format)
    tn3 = time.time()
    print(f'BufferALloc Time: {tn3-tn2}')
    # process the segmentation network
    net.Process(img_input, ignore_class=args.ignore_class)
    tn4 = time.time()
    print(f'PROCESS SEGMENTATION MASK TIME: {tn4-tn3}')
    # generate the overlay
    #if buffers.overlay:
    #    net.Overlay(buffers.overlay, filter_mode=args.filter_mode)
    
    # generate the mask
    if buffers.mask:
        net.Mask(buffers.mask, filter_mode=args.filter_mode)
    tn5 = time.time()
    print(f'GENERATE MASK TIME: {tn5-tn4}')
    # Coverting CUDA image to numpy ndarray
    
    tensor = torch.as_tensor(buffers.mask, device='cuda')
    torch_mask = buffers.mask
    tn6 = time.time()
    print(f'NUMPY MASK Time: {tn6 - tn5}')
    ton = time.time()
    print(f'FPS WITHOUT NUMPY OPS: {ton - t1}')
    
    height, width, _ = torch_mask.shape
    
    nops1 = time.time()
    # Defining target RGB values for Road Class and Footpath Class as PyTorch tensors
    target_class_road = torch.tensor([128, 64, 128], dtype=torch.uint8)
    target_class_footpath = torch.tensor([150, 75, 200], dtype=torch.uint8)
    nops2 = time.time()
    print(f'Define RGB arrays: {nops2 - nops1}')
    
    # Defining and extracting ROI - Road
    roi_top_road = height // 2
    roi_bottom_road = height
    roi_road = torch_mask[roi_top_road:roi_bottom_road, :]
    mask_road = torch.all(roi_road == target_class_road, dim=-1)
    nops3 = time.time()
    print(f'Define and extract ROI: {nops3 - nops2}')
    
    # Defining and extracting ROI - Footpath
    roi_height_fp = height // 4
    roi_width_fp = width // 2
    roi_top_fp = height - roi_height_fp
    roi_bottom_fp = height
    roi_left_fp = width // 4
    roi_right_fp = 3 * (width // 4)
    roi_fp = torch_mask[roi_top_fp:roi_bottom_fp, roi_left_fp:roi_right_fp]
    mask_fp = torch.all(roi_road == target_class_footpath, dim=-1)
    nops4 = time.time()
    print(f'Define and extract ROI: {nops4 - nops3}')
    
    # Calculating Overlap percentage - Road
    total_pixels_in_roi_road = roi_road.shape[0] * roi_road.shape[1]
    total_road_pixels = mask_road.sum().item()
    overlap_percent_road = (total_road_pixels / total_pixels_in_roi_road) * 100
    nops5 = time.time()
    print(f'Calculate Overlap: {nops5 - nops4}')
    
    # Calculating Overlap percentage - Footpath
    total_pixels_in_roi_fp = roi_fp.shape[0] * roi_fp.shape[1]
    total_fp_pixels = mask_fp.sum().item()
    overlap_percent_fp = (total_fp_pixels / total_pixels_in_roi_fp) * 100
    nops6 = time.time()
    print(f'Calculate Overlap: {nops6 - nops5}')
    
    print(f'Road Overlap % is: {overlap_percent_road}')
    print(f'Footpath Overlap % is: {overlap_percent_fp}')
    tn7 = time.time()
    print(f'Numpy OPS Time: {tn7 - tn6}')
    
    # composite the images
    #if buffers.composite:
    #    cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
    #    cudaOverlay(buffers.mask, buffers.composite, buffers.overlay.width, 0)

    # render the output image
    #output.Render(buffers.output)
    t2 = time.time()
    print(f'FPS observed: {1/(t2-t1)}   {t2-t1}')

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    cudaDeviceSynchronize()
    #net.PrintProfilerTimes()

    # compute segmentation class stats
    if args.stats:
        buffers.ComputeStats()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
    full = time.time()
    print(f"1 iteration fps = {1/(full - t1)} {full}  {t1}")
complete = time.time()
print(f'TIME TAKEN: {complete - load_time1}')
