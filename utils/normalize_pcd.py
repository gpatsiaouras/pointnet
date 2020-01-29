# Example
# python3 utils/normalize_pcd.py --folder data/giorgos_test_pcd_files/ --filename_in model_2048.pcd --filename_out model_2048_normalized.pcd

import pcl
import numpy as np
from glob import glob
import argparse


def normalize(pointcloud_in):
    max = 0.36
    min = -0.36

    pc_in = np.asarray(pointcloud_in, dtype=np.float32)

    flatten_pc = pc_in.flatten()
    multiplier = (flatten_pc - flatten_pc.min()) / (flatten_pc.max() - flatten_pc.min())

    pc_out_array = multiplier.reshape(2048, 3) * (max - min) + min
    pointcloud_out = pcl.PointCloud()
    pointcloud_out.from_array(pc_out_array)

    return pointcloud_out


def normalize_and_save(file_in, file_out):
    pointcloud_in = pcl.load(file_in)
    pointcloud_out = normalize(pointcloud_in)
    pcl.save(pointcloud_out, file_out)


def batch_normalize(pointclouds_in):
    pointclouds_out = []
    for pointcloud in pointclouds_in:
        pointclouds_out.append(normalize(pointcloud))

    return pointclouds_out


def normalize_folder_and_save(pcd_folder, filename_in, filename_out):
    pcd_folders = glob(pcd_folder + "*/")
    for folder in pcd_folders:
        normalize_and_save(folder + filename_in, folder + filename_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_in', required=True, help='Folder that contains pcd files With trailing slash')
    parser.add_argument('--filename_in', required=True, help='Name of the file to read a pcd')
    parser.add_argument('--filename_out', required=True, help='Name of the file to write the normalized pcd')
    FLAGS = parser.parse_args()

    normalize_folder_and_save(FLAGS.folder_in, FLAGS.filename_in, FLAGS.filename_out)
