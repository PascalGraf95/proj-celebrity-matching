from tensorflow.keras.applications.inception_v3 import preprocess_input
import tensorflow as tf
import os, json
from tqdm import tqdm
import numpy as np
import argparse


DATA_FOLDER = "./Supportset_Celebrities_crop"
DATABASE = "./database_celebrities_crop.json"
MODEL_NAME = "best_encoder_505000_step_acc_0_9013.h5"#"best_encoder_419000_step_ap_0_2872_an_1_8216.h5"
FILE_AMOUNT = 7

encoder = tf.keras.models.load_model(os.path.join('vgg_models', MODEL_NAME), compile=False)


def read_image(path):
    image = tf.keras.preprocessing.image.load_img(path,
                                                  color_mode="rgb",
                                                  target_size=(300, 300),
                                                  interpolation="bilinear")
    image = tf.keras.preprocessing.image.img_to_array(image, dtype='float32')
    return image


def main(remove_obsolete_folders):
    folder_list = []
    feature_list = []
    mean_feature_list = []
    representative_data_points = []
    file_list = []
    obsolete_folders = []

    for folder in tqdm([folder for folder in os.listdir(DATA_FOLDER) if os.path.isdir(os.path.join(DATA_FOLDER, folder))]):
        tmp_file_names = []

        files = os.listdir(os.path.join(DATA_FOLDER, folder))
        number_of_images = min(len(files), FILE_AMOUNT)

        if number_of_images < FILE_AMOUNT:
            obsolete_folders.append(folder)
            print(f"WARNING: {folder} has only {len(files)} images")

            if remove_obsolete_folders:
                os.remove(os.path.join(DATA_FOLDER, folder))
            continue

        image_batch = np.zeros((number_of_images, 300, 300, 3))
        for idx, file in enumerate(files[:number_of_images]):
            image_batch[idx] = read_image(os.path.join(DATA_FOLDER, folder, file))
            tmp_file_names.append(file)

        batch_features = encoder(preprocess_input(image_batch)).numpy()
        mean_feature = np.mean(batch_features, axis=0)
        # Calculate Euclidean distances from each point to the mean
        distances = np.linalg.norm(batch_features - mean_feature, axis=1)
        # Find the index of the point with the minimum distance
        closest_point_index = np.argmin(distances)
        # Get the closest point
        closest_point = batch_features[closest_point_index]

        # Append to lists
        mean_feature_list.append(mean_feature.tolist())
        representative_data_points.append(closest_point.tolist())
        folder_list.append(folder)
        feature_list.append(batch_features.tolist())
        file_list.append(tmp_file_names)
    database = {"folders": folder_list, "features": feature_list, "mean_features": mean_feature_list,
                "representative_features": representative_data_points, "files": file_list}

    with open(DATABASE, 'w') as f:
        json.dump(database, f)

    with open("obsolete_folders.txt", 'w', encoding="utf-8") as f:
        for item in obsolete_folders:
            f.write(item + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process images and optionally delete faulty images.")
    parser.add_argument('-rem', '--remove_obsolete_folders', type=bool, nargs="?", default=False,
                        help="Delete faulty images (True or False)", required=False)

    args = parser.parse_args()
    main(args.remove_obsolete_folders)
