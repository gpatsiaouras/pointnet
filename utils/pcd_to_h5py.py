import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from glob import glob
import pcl
import shutil
import progressbar
from data_prep_util import save_h5
import numpy as np
import random

def check_2048_and_prompt_for_delete(pcd_folders):
    to_be_deleted = []
    counter = 0
    print("Checking pcd files for pointclouds with 2048 exact points....")
    for pcd_folder in progressbar.progressbar(pcd_folders):
        pointcloud = pcl.load(pcd_folder + "/model_2048.pcd")
        if pointcloud.size != 2048:
            to_be_deleted.append(pcd_folder)
        counter += 1

    if len(to_be_deleted) > 0:
        decision = input("There are " + str(len(
            to_be_deleted)) + " folders that contain pointclouds with less "
                              "points than 2048\nShould I delete these folders (y/n)? ")
        if decision == "y":
            [shutil.rmtree(path) for path in to_be_deleted]
            print("Successfully deleted %d pointclouds" % (len(to_be_deleted)))
        else:
            print("Operation aborted")
    else:
        print("All pointclouds are 2048 points!")


def load_pointclouds(pcd_folders):
    print("Loading pointclouds....")
    pointclouds = []
    for pcd_folder in progressbar.progressbar(pcd_folders):
        pointclouds.append(pcl.load(pcd_folder + "/model_2048.pcd"))
    print("%d pointclouds loaded successfully" % len(pointclouds))

    return pointclouds


def get_data_from_pointclouds(size, airplane_pointclouds, car_pointclouds):
    data = np.zeros((size, 2048, 3))
    label = np.zeros((size, 1), dtype=np.int)

    enumerator = 0
    for i in range(size):
        if enumerator < len(airplane_pointclouds):
            data[i] = np.asarray(airplane_pointclouds[i])
            label[i] = 0
        else:
            data[i] = np.asarray(car_pointclouds[i - len(airplane_pointclouds)])
            label[i] = 1
        enumerator += 1
    return data, label


def get_h5_from_pcd(pcd_path, file_name):
    pointcloud = pcl.load(pcd_path)
    data = np.zeros((1, pointcloud.size, 3))
    label = np.array([0], dtype=np.int)
    data[0] = np.asarray(pointcloud)
    save_h5(file_name, data, label, "float64")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", default="", help="Specify the path to the input pointcloud pcd")
    group.add_argument("--folder", default="", help="Specify folder containing pointcloud pcd files")
    FLAGS = parser.parse_args()

    if FLAGS.file != "":
        folder_to_save = "data/giorgos/"
        filename = folder_to_save + str(random.randint(0, 100)) + "_pcd_2048.h5"
        get_h5_from_pcd(FLAGS.file, filename)
    else:
        # We have 2913 airplanes and 731 cars
        airplane_path = "/home/giorgos/Public/shapenet/2048_res_pcd_dataset/airplane-02691156/*"
        car_path = "/home/giorgos/Public/shapenet/2048_res_pcd_dataset/car-02958343/*"

        # Assess that all pcd files have exactly 2048 points before performing any operation
        check_2048_and_prompt_for_delete(glob(airplane_path) + glob(car_path))

        # Get pcd folders for this operation
        car_pcd_folders = glob(car_path)
        airplane_pcd_folders = glob(airplane_path)

        # Create pcl objects for each pcd file
        print("Creating pointcloud car objects ")
        car_pointclouds = load_pointclouds(car_pcd_folders)
        print("Creating pointcloud airplane objects ")
        airplane_pointclouds = load_pointclouds(airplane_pcd_folders)

        # Create two batches of data
        data0, label0 = get_data_from_pointclouds(2048, airplane_pointclouds[:1648], car_pointclouds[:400])
        data1, label1 = get_data_from_pointclouds(1024, airplane_pointclouds[1648:2496], car_pointclouds[400:600])

        # Call util to write h5 file
        save_h5("data/giorgos/train0.h5", data0, label0, "float64")
        save_h5("data/giorgos/test0.h5", data1, label1, "float64")

    print("\n Completed!")
