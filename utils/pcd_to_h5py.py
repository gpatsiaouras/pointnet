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
import normalize_pcd


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


def load_pointclouds(pcd_folders, pcd_filename):
    print("Loading pointclouds....")
    pointclouds = []
    for pcd_folder in progressbar.progressbar(pcd_folders):
        pointclouds.append(pcl.load(pcd_folder + "/" + pcd_filename))
    print("%d pointclouds loaded successfully" % len(pointclouds))

    return pointclouds


def get_data_from_pointclouds(size, zero_label_pointclouds, first_label_pointclouds):
    data = np.zeros((size, 2048, 3))
    label = np.zeros((size, 1), dtype=np.int)

    enumerator = 0
    for i in range(size):
        if enumerator < len(zero_label_pointclouds):
            data[i] = np.asarray(zero_label_pointclouds[i])
            label[i] = 0
        else:
            data[i] = np.asarray(first_label_pointclouds[i - len(zero_label_pointclouds)])
            label[i] = 1
        enumerator += 1
    return data, label


def create_h5_from_file(pcd_path, file_name):
    # Load a single pointcloud
    pointcloud = pcl.load(pcd_path)

    # Normalize the pointcloud
    normalized_pointcloud = normalize_pcd.normalize(pointcloud)

    # Create a data array to hold the point data
    data = np.zeros((1, normalized_pointcloud.size, 3))

    # Create a label array to indicate label for this data table
    label = np.array([0], dtype=np.int)

    # Put pointcloud inside the data array
    data[0] = np.asarray(normalized_pointcloud)

    # Write array to h5 file
    save_h5(file_name, data, label, "float64")


def create_h5_from_folder(pcd_folder, h5_filename):
    # Get the folders containing the pcd files
    files_in_folder = glob(pcd_folder + "*")

    # Create pointcloud objects out of the models inside the folders with name model_2048_normalized.pcd
    pointclouds = load_pointclouds(files_in_folder, "model_2048.pcd")

    # Normalize the pointclouds before loading
    normalized_pointclouds = normalize_pcd.batch_normalize(pointclouds)

    # Create the numpy arrays to be put inside the h5file
    data0, label0 = get_data_from_pointclouds(len(files_in_folder),
                                              [np.asarray(norm_poin) for norm_poin in normalized_pointclouds], [])

    # Save the file to an h5file
    save_h5(h5_filename, data0, label0, "float64")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file_in", default="", help="Specify the path to the input pointcloud pcd")
    group.add_argument("--folder_in", default="",
                       help="Specify folder containing pointcloud pcd files with trailing slash")
    parser.add_argument("--h5filename", default="data/classification/customtest0.h5",
                        help="Specify the name of the pcd files of the output")
    FLAGS = parser.parse_args()

    if FLAGS.file_in != "":
        create_h5_from_file(FLAGS.file_in, FLAGS.h5filename)
    elif FLAGS.folder_in != "":
        create_h5_from_folder(FLAGS.folder_in, FLAGS.h5filename)
    else:
        # We have 2913 airplanes and 731 cars
        airplane_path = "/home/classification/Public/shapenet/2048_res_pcd_dataset/airplane-02691156/*"
        car_path = "/home/classification/Public/shapenet/2048_res_pcd_dataset/car-02958343/*"

        # Assess that all pcd files have exactly 2048 points before performing any operation
        check_2048_and_prompt_for_delete(glob(airplane_path) + glob(car_path))

        # Get pcd folders for this operation
        car_pcd_folders = glob(car_path)
        airplane_pcd_folders = glob(airplane_path)

        # Create pcl objects for each pcd file
        print("Creating pointcloud car objects ")
        car_pointclouds = load_pointclouds(car_pcd_folders, "model_2048.pcd")
        print("Creating pointcloud airplane objects ")
        airplane_pointclouds = load_pointclouds(airplane_pcd_folders, "model_2048.pcd")

        # Create two batches of data
        data0, label0 = get_data_from_pointclouds(2048, airplane_pointclouds[:1648], car_pointclouds[:400])
        data1, label1 = get_data_from_pointclouds(1024, airplane_pointclouds[1648:2496], car_pointclouds[400:600])

        # Call util to write h5 file
        save_h5("data/classification/train0.h5", data0, label0, "float64")
        save_h5("data/classification/test0.h5", data1, label1, "float64")

    print("\n Completed!")
