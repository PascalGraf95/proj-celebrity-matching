from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QScrollArea
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QMovie
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import tensorflow as tf
import numpy as np
import os
import cv2
import sys
import json
import face_recognition
import matplotlib.pyplot as plt

CROP_IMAGES = True
FEATURE_TYPE = "mean"  # mean, all or representative
N_SAMPLES = 10
N_MATCHES = 3
N_PCA = N_MATCHES
WEBCAM_RESOLUTION = (1080, 720)
MODEL_NAME = "best_encoder_505000_step_acc_0_9013.h5"#"best_encoder_419000_step_ap_0_2872_an_1_8216.h5"
SUPPORT_SET_PATH = "./Supportset_Celebrities_crop"
DATABASE = "./database_celebrities_crop.json"


encoder = tf.keras.models.load_model(os.path.join('vgg_models', MODEL_NAME), compile=False)


def get_ranking(dist):
    if dist < 0.1:
        return "Ähnlichkeit: Sehr hoch (Virt. Distanz: {:.2f})".format(dist)
    if dist < 0.2:
        return "Ähnlichkeit: Hoch (Virt. Distanz: {:.2f})".format(dist)
    if dist < 0.3:
        return "Ähnlichkeit: Mittel (Virt. Distanz: {:.2f})".format(dist)
    if dist < 0.6:
        return "Ähnlichkeit: Gering (Virt. Distanz: {:.2f})".format(dist)
    if dist < 1:
        return "Ähnlichkeit: Sehr gering (Virt. Distanz: {:.2f})".format(dist)
    else:
        f"Keine ähnliche Person konnte im Datenset gefunden werden."


def read_image(path):
    image = tf.keras.preprocessing.image.load_img(path,
                                                  color_mode="rgb",
                                                  target_size=(300, 300),
                                                  interpolation="bilinear")

    image = tf.keras.preprocessing.image.img_to_array(image, dtype='float32')
    return image


def get_pca(features, folders):
    pca = PCA(n_components=2)
    z = pca.fit_transform(features)

    fig = plt.figure()
    fig.add_subplot(111)

    df = pd.DataFrame()
    df["y"] = folders
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", len(list(set(folders)))),
                    data=df, alpha=0.6).set(title=f"PCA")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.tight_layout()

    ## Plot to np.array
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    ##

    return data


def process_recorded_images(image_buffer):
    with open(DATABASE, "r") as f:
        database = json.load(f)
    dataset_folder_list = database["folders"]
    mean_dataset_feature_list = database["mean_features"]
    representative_dataset_feature_list = database["representative_features"]
    dataset_feature_list = database["features"]
    dataset_file_list = database["names"]

    ### Create features of webcam input
    if CROP_IMAGES:
        input_images = []
        for image in image_buffer:
            image = np.array(image)
            face_locations = face_recognition.face_locations(image)
            if len(face_locations) > 0:
                # Define crop region
                y_start = face_locations[0][0]
                y_end = face_locations[0][2]
                x_start = face_locations[0][3]
                x_end = face_locations[0][1]
                add_length = int((x_end - x_start))//2
                new_y_start = max(y_start - add_length, 0)
                new_y_end = min(y_end + add_length, image.shape[0])
                new_x_start = max(x_start - add_length, 0)
                new_x_end = min(x_end + add_length, image.shape[1])

                cropped_image = image[new_y_start:new_y_end, new_x_start:new_x_end]
                input_images.append(cv2.resize(cropped_image, (300, 300), interpolation=cv2.INTER_LINEAR))
        input_images = np.array(input_images)
    else:
        input_images = np.array([cv2.resize(image, (300, 300), interpolation=cv2.INTER_LINEAR) for image in image_buffer])

    if not len(input_images):
        return None, None, None, None, None

    target_features = encoder(preprocess_input(input_images))
    mean_target_features = np.mean(target_features, axis=0)

    distances = []
    if FEATURE_TYPE == "mean":
        # Iterate through all persons in the support dataset
        for dataset_person_features in mean_dataset_feature_list:
            # Calculate the distance between each person's mean features and the target person's mean features
            dist = tf.reduce_sum((mean_target_features - dataset_person_features)**2, axis=-1).numpy()
            alt_dist = np.linalg.norm(mean_target_features - dataset_person_features, axis=1)
            distances.append(dist)

    elif FEATURE_TYPE == "representative":
        # Iterate through all persons in the support dataset
        for dataset_person_features in representative_dataset_feature_list:
            # Calculate the distance between each person's mean features and the target person's mean features
            dist = tf.reduce_sum((mean_target_features - dataset_person_features)**2, axis=-1).numpy()
            alt_dist = np.linalg.norm(mean_target_features - dataset_person_features, axis=1)
            distances.append(dist)

    elif FEATURE_TYPE == "all":
        # Iterate through all persons in the support dataset
        for idx, (dataset_person_folder, dataset_person_features, dataset_person_files) \
                in enumerate(zip(dataset_folder_list, dataset_feature_list, dataset_file_list)):

            # ToDo: Store the respective image along with the smallest distance for each person.
            person_distances = []
            # person_files = []
            # Iterate through all available images for each person
            for person_feature, person_file in zip(dataset_person_features, dataset_person_files):
                dist = tf.reduce_sum((mean_target_features - person_feature)**2, axis=-1).numpy()
                person_distances.append(dist)
                # person_files.append(person_file)
            # Store the smallest distance along with the respective file path
            distances.append(np.min(person_distances))

    # Sort the dataset by distance to the target person
    sorted_indices = np.argsort(distances)
    sorted_distances = [distances[i] for i in sorted_indices]
    sorted_dataset_folder_list = [dataset_folder_list[i] for i in sorted_indices]
    sorted_dataset_feature_list = [dataset_feature_list[i] for i in sorted_indices]
    sorted_dataset_file_list = [dataset_file_list[i] for i in sorted_indices]

    final_file_list = []
    final_distance_list = []
    # Iterate through the closest person's images and extracted features and find the actual closest image to the
    # target person.
    for person_features, person_files in zip(sorted_dataset_feature_list[:N_MATCHES],
                                             sorted_dataset_file_list[:N_MATCHES]):
        person_distances = []
        for feature in person_features:
            dist = tf.reduce_sum((mean_target_features - feature)**2, axis=-1).numpy()
            person_distances.append(dist)
        smallest_distance_index = np.argsort(distances)[0]

        final_distance_list.append(distances[smallest_distance_index])
        final_file_list.append(person_files[smallest_distance_index])

    pca_folders = [*["me"]*len(input_images)]
    features = []
    for idx in range(N_PCA):
        folder = sorted_dataset_folder_list[idx]
        pca_folders.extend([f"Close example: {folder}"]*len(sorted_dataset_feature_list[idx]))
        features.append(sorted_dataset_feature_list[idx])
    for _ in range(5):
        idx = np.random.randint(low=N_PCA, high=len(sorted_dataset_folder_list))
        folder = sorted_dataset_folder_list[idx]
        pca_folders.extend([f"Random example: {folder}"]*len(sorted_dataset_feature_list[idx]))
        features.append(sorted_dataset_feature_list[idx])
    features = np.array(features)
    features = features.reshape(features.shape[0]*features.shape[1], features.shape[-1])
    pca_features = [*target_features, *features]

    return final_file_list, sorted_dataset_folder_list[:N_MATCHES], sorted_distances, \
        pca_features, pca_folders


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.showFullScreen()

        self.image_label = QLabel()
        self.record_button = QPushButton("Match")
        self.reset_button = QPushButton('Reset')

        self.movie = QMovie("./loader.gif")

        self.is_recording = False

        self.image_buffer = []

        self.init_ui()
        self.init_camera()

    def init_ui(self):
        layout = QVBoxLayout()

        self.setStyleSheet('font-size: ' + str(22)+'px')

        self.image_label = QLabel()

        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.record_button)
        #button_layout.addWidget(self.reset_button)

        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        #self.setFixedSize(QSize(*WINDOW_SIZE))
        self.setWindowTitle("Celebrity Matching")

        self.record_button.clicked.connect(self.start_recording)
        self.reset_button.clicked.connect(self.reset)

    def init_camera(self):
        # Initialize the camera capture
        self.camera = cv2.VideoCapture(0)  # Use the default camera

        # Create a timer to continuously update the camera image
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)  # Update every 30 milliseconds (33 fps)

    def update_camera(self):
        ret, frame = self.camera.read()  # Read a frame from the camera

        if ret:
            # Convert the frame to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.flip(frame_rgb, 1)
            frame_rgb = np.array(frame_rgb[0:1080, int((WEBCAM_RESOLUTION[0]/2)-(WEBCAM_RESOLUTION[1]/2)):int((WEBCAM_RESOLUTION[0]/2)+(WEBCAM_RESOLUTION[1]/2))]) # TODO: Webcam Resolution

            # Create a QImage from the frame
            image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format.Format_RGB888)

            # Display the image on the QLabel
            pixmap = QPixmap.fromImage(image)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

            if self.is_recording:
                # Add the image to the buffer
                self.image_buffer.append(frame_rgb)

                if len(self.image_buffer) >= N_SAMPLES:
                    
                    self.record_button.setText('Done')
                    self.timer.stop()
                    self.image_label.setMovie(self.movie)
                    self.movie.start()
                    self.image_label.show()
                    self.is_recording = False
                    #self.loading_spinner()
                    self.show_result()

    def start_recording(self):
        self.record_button.setText('Recording')
        self.record_button.setDisabled(True)
        self.is_recording = True

    def loading_spinner(self):
        self.image_label.setMovie(self.movie)
        self.movie.start()
        self.image_label.show()

    def stop_loading_spinner(self):
        self.movie.stop()

    def reset(self):
        self.is_recording = False
        self.image_buffer = []
        self.record_button = QPushButton("Match")
        self.init_ui()
        self.init_camera()

    def show_result(self):
        # files, folders, distances, feature_list, current_folders = process_recorded_images(self.image_buffer)
        file_list, folder_list, distances, pca_features, pca_folders = process_recorded_images(self.image_buffer)
        self.stop_loading_spinner()

        layout = QVBoxLayout()

        if file_list is None:
            label = QLabel("No face has been detected")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)

        else:
            image = self.image_buffer[0]
            image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            pixmaps = [pixmap]
            for i in range(N_MATCHES):
                pixmaps.append(QPixmap(os.path.join(SUPPORT_SET_PATH,
                                                    folder_list[i],
                                                    file_list[i])))

            for i, pixmap in enumerate(pixmaps):
                if i >= 1:
                    label = QLabel(f"{folder_list[i-1]} {get_ranking(distances[i-1])}")
                    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    layout.addWidget(label)

                image_label = QLabel()
                image_label.setPixmap(pixmap.scaled(image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
                image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(image_label)

            image = get_pca([*pca_features], pca_folders)
            image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            image_label = QLabel()
            image_label.setPixmap(pixmap.scaled(image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(image_label)

        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset)
        layout.addWidget(self.reset_button)
            
        container = QWidget()
        container.setLayout(layout)

        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.setCentralWidget(scroll)
            


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('QtCurve')

    window = MainWindow()
    window.show()

    app.exec()