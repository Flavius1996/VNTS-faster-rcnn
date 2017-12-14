#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick - Modified by Hoang Huu Tin
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import glob as glob

CLASSES = ('__background__',  # always index 0
                '102', '103a', '103b', '104', '105', '112', '123a', '123b', '124a', '127_40', '127_50', '127_60', '127_80', 
                 '130', '131a', '205d', '207b', '207c', '208', '224', '225', '227', '239', '245a', '302a', '305', '407a', '412b')

NETS = {'vgg16': ('VGG16',
                  'vgg16_faster_rcnn_iter_20000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                  'vgg_cnn_m_1024_faster_rcnn_iter_60000.caffemodel'),
        'zf': ('ZF',
                  'zf_faster_rcnn_iter_70000.caffemodel')}

#        'zf': ('ZF',
#                 'zf_faster_rcnn_iter_20000.caffemodel')}


def vis_detections(im_file, class_name, dets, ax, inds):
    """Draw detected bounding boxes."""

    #im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=1.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.2),
                fontsize=14, color='white')

    #plt.axis('off')
    #plt.tight_layout()
    #plt.draw()
    #if save_img:
    #    plt.savefig(im_file, bbox_inches='tight')

def demo(net, image_name, confidence_threshold):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = confidence_threshold

    NMS_THRESH = 0.3
    im_path = os.path.join(cfg.DATA_DIR, 'demo', 'result')
    if not os.path.exists(im_path):
        os.makedirs(im_path)
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.imshow(im, aspect='equal')

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        im_name = image_name.split('/')[-1].replace('.ppm','.jpg')
        im_path_ = os.path.join(im_path, im_name)

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) != 0:
            vis_detections(im_path_, cls, dets, ax, inds)
            
            # Only save images have detected signs

            #plt.axis('off')
            #plt.tight_layout()
            #plt.draw()
            #plt.savefig(im_path_, bbox_inches='tight')


    # Save all images (detected and no-detected sign)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(im_path_, bbox_inches='tight')


    plt.close()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='zf')

    parser.add_argument('--confidence', dest='confidence_threshold', help='Confidence Threshold',
                        default=0.5)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'VNTSDB','TrainedModel',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR,'demo','*.ppm'))
    print 'Number of Images: {}'.format(len(im_names))
    print 'Confidence Threshold: {}'.format(args.confidence_threshold)
    count = 0
    for im_name in im_names:
        count += 1
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print '{}. Image data/demo/{}'.format(count, im_name)
        demo(net, im_name, float(args.confidence_threshold))

    #plt.show()

