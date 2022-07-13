#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
from utils.laserscan import LaserScan, SemLaserScan
from utils.laserscanvis import LaserScanVis

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./visualize.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=False,
        default="/home/mayq/data1/datasets/WADS/val",
        help='Dataset to visualize. No Default',
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default="config/semantic-WADS.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        default="/home/mayq/data1/lidar_denoise/result/86.4/",
        required=False,
        help='Alternate location for labels, to use predictions folder. '
             'Must point to directory containing the predictions in the proper format '
             ' (see readme)'
             'Defaults to %(default)s',
    )
    parser.add_argument(
        '--ignore_semantics', '-i',
        dest='ignore_semantics',
        default=False,
        action='store_true',
        help='Ignore semantics. Visualizes uncolored pointclouds.'
             'Defaults to %(default)s',
    )
    parser.add_argument(
        '--noisy_labels', '-n',
        type=list,
        default=[110, 111],
        help='Noisy labels such as rain, snow, fog and so on...',
    )
    parser.add_argument(
        '--pred_noisy_labels', '-pn',
        type=list,
        default=[2],
        help='Predict noisy labels in prediction result file.',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='86.4',
        help='Model used to output prediction results.',
    )
    # parser.add_argument(
    #     '--offset',
    #     type=int,
    #     default=0,
    #     required=False,
    #     help='Sequence to start. Defaults to %(default)s',
    # )
    parser.add_argument(
        '--show_clear',
        default=True,
    )
    parser.add_argument(
        '--show_denoised',
        default=True,
    )
    parser.add_argument(
        '--ignore_safety',
        dest='ignore_safety',
        default=False,
        action='store_true',
        help='Normally you want the number of labels and ptcls to be the same,'
             ', but if you are not done inferring this is not the case, so this disables'
             ' that safety.'
             'Defaults to %(default)s',
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Dataset", FLAGS.dataset)
    print("Config", FLAGS.config)
    print("Predictions", FLAGS.predictions)
    print("ignore_semantics", FLAGS.ignore_semantics)
    print("ignore_safety", FLAGS.ignore_safety)
    print("*" * 80)

    # open config file
    try:
        print("Opening config file %s" % FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    # does dataset folder exist?
    scan_paths = os.path.join(FLAGS.dataset, "velodyne")
    if os.path.isdir(scan_paths):
        print("dataset folder exists! Using velodyne folder %s" % scan_paths)
    else:
        print("dataset folder doesn't exist! Exiting...")
        quit()

    # populate the pointclouds
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort()

    # does sequence folder exist?
    if not FLAGS.ignore_semantics:
        if FLAGS.predictions is not None:
            pred_paths = FLAGS.predictions
        label_paths = os.path.join(FLAGS.dataset, "labels")
        if os.path.isdir(label_paths):
            print("Labels folder exists! Using labels from %s" % label_paths)
        else:
            print("Labels folder doesn't exist! Exiting...")
            quit()
        # populate the pointclouds
        label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(label_paths) for f in fn]
        label_names.sort()
        prelabel_names = None
        if FLAGS.show_denoised:
            prelabel_paths = os.path.join('result', FLAGS.model_name)
            prelabel_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(prelabel_paths) for f in fn]
            prelabel_names.sort()

        # check that there are same amount of labels and scans
        if not FLAGS.ignore_safety:
            assert (len(label_names) == len(scan_names))

    # create a scan
    if FLAGS.ignore_semantics:
        scan = LaserScan(project=True)  # project all opened scans to spheric proj
    else:
        color_dict = CFG["color_map"]
        nclasses = len(color_dict)
        scan = SemLaserScan(nclasses, color_dict, project=True, noise_labels=FLAGS.noisy_labels,
                            pred_noise=FLAGS.pred_noisy_labels, show_clear=FLAGS.show_clear, show_denoised=FLAGS.show_denoised)

    # create a visualizer
    semantics = not FLAGS.ignore_semantics
    if not semantics:
        label_names = None
    vis = LaserScanVis(scan=scan,
                       scan_names=scan_names,
                       label_names=label_names,
                       prelabel_names=prelabel_names,
                       semantics=semantics,
                       show_clear=FLAGS.show_clear,
                       show_denoised=FLAGS.show_denoised)

    # print instructions
    print("To navigate:")
    print("\tb: back (previous scan)")
    print("\tn: next (next scan)")
    print("\tq: quit (exit program)")

    # run the visualizer
    vis.run()
