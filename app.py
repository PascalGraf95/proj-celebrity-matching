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
MEAN_FEATURES = True
N_SAMPLES = 10
N_MATCHES = 3
N_PCA = N_MATCHES
WEBCAM_RESOLUTION = (1080, 720)
MODEL_NAME = "best_encoder_505000_step_acc_0_9013.h5"#"best_encoder_419000_step_ap_0_2872_an_1_8216.h5"
SUPPORT_SET_PATH = "./Supportset_Celebrities_crop"
DATABASE = "./database_celebrities_crop.json"

def get_ranking(dist):
    dist = round(dist, 4)
    if dist<0.1:
        return f"Sehr ähnlich mit Abstand: {dist}"
    if dist<0.2:
        return f"Ähnlich mit Abstand: {dist}"
    if dist<0.3:
        return f"Bisschen ähnlich mit Abstand: {dist}"
    if dist<0.6:
        return f"Wenig Ähnlichkeit mit Abstand: {dist}"
    if dist<1:
        return f"Sehr wenig Ähnlichkeit mit Abstand: {dist}"
    else:
        f"Keine Ähnlchkeit konnte im Datenset gefunden werden. Abstand: {dist}"

def read_image(path):
    image = tf.keras.preprocessing.image.load_img(path,
                                                  color_mode="rgb",
                                                  target_size=(300, 300),
                                                  interpolation="bilinear")

    image = tf.keras.preprocessing.image.img_to_array(image, dtype='float32')
    return image

encoder = tf.keras.models.load_model(os.path.join('vgg_models', MODEL_NAME), compile=False)

def get_PCA(features, folders):

    pca = PCA(n_components=2)
    z = pca.fit_transform(features)

    #tsne = TSNE(n_components=2, perplexity=10,learning_rate='auto', init='pca', verbose=0, random_state=123)
    #z = tsne.fit_transform(np.array(features))

    fig = plt.figure()
    fig.add_subplot(111)

    df = pd.DataFrame()
    df["y"] = folders
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

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

def process_recorded_images(image_buffer, mean=MEAN_FEATURES):

    with open(DATABASE, "r") as f:
        database = json.load(f)
    folder_list = database["folders"]
    mean_feature_list = database["mean_features"]
    feature_list = database["features"]
    name_list = database["names"]

    ### Create features of webcam input
    if CROP_IMAGES:
        input_images = []
        for image in image_buffer:
            image = np.array(image)
            face_locations = face_recognition.face_locations(image)
            if len(face_locations)>0:
                #crop = image[face_locations[0][0]:face_locations[0][2], face_locations[0][3]:face_locations[0][1]]
                ###
                y_start = face_locations[0][0]
                y_end = face_locations[0][2]
                x_start = face_locations[0][3]
                x_end = face_locations[0][1]

                done = False
                start = 0.5
                decrease = 0.01
                while not done:
                    add_length = int((x_end - x_start)*start)//2
                    new_y_start = y_start - add_length
                    new_y_end = y_end + add_length
                    new_x_start = x_start - add_length
                    new_x_end = x_end + add_length
                    if new_y_start>=0 and new_x_start>=0 and new_y_end<=image.shape[0] and new_x_end<=image.shape[1]:
                        done = True
                        break
                    start -= decrease

                crop = image[new_y_start:new_y_end, new_x_start:new_x_end]
                ###
                input_images.append(cv2.resize(crop, (300, 300), interpolation = cv2.INTER_AREA))
        input_images = np.array(input_images)
    else:
        input_images = np.array([cv2.resize(image, (300, 300), interpolation = cv2.INTER_AREA) for image in image_buffer])

    if len(input_images)<1:
        return None, None, None, None, None

    own_features = encoder(preprocess_input(input_images))
    rep = np.mean(own_features, axis=0)
    ###

    if mean:
        mean_distances = []
        for features in mean_feature_list:
            dist = tf.reduce_sum((rep - features)**2, axis=-1).numpy()
            mean_distances.append(dist)

        idx = np.argsort(mean_distances)
        mean_distances = [mean_distances[i] for i in idx]
        folder_list = [folder_list[i] for i in idx]
        feature_list = [feature_list[i] for i in idx]
        name_list = [name_list[i] for i in idx]

        final_files = []
        final_folders = []
        final_distances = []
        for j in range(N_MATCHES):
            if j<len(feature_list):
                distances = []
                features = feature_list[j]
                folder = folder_list[j]
                names = name_list[j]
                for feature in features:
                    dist = tf.reduce_sum((rep - feature)**2, axis=-1).numpy()
                    distances.append(dist)
                idx = np.argsort(distances)
                distances = [distances[i] for i in idx]
                names = [names[i] for i in idx]
                final_distances.append(distances[0])
                final_folders.append(folder)
                final_files.append(names[0])

    else:
        final_distances = []
        final_files = []
        final_folders = []
        final_features = []
        mean_distances = []

        for i, folder in enumerate(folder_list):
            features = feature_list[i]
            names = name_list[i]

            current_distances = []
            current_names = []
            for feature, name in zip(features, names):
                dist = tf.reduce_sum((rep - feature)**2, axis=-1).numpy()
                current_distances.append(dist)
                current_names.append(name)

            idx = np.argsort(current_distances)
            current_distances = [current_distances[j] for j in idx]
            current_names = [current_names[j] for j in idx]
            final_distances.append(current_distances[0])
            mean_distances.append(np.mean(current_distances))
            final_files.append(current_names[0])
            final_folders.append(folder)
            final_features.append(features)

        idx = np.argsort(final_distances)
        final_distances = [final_distances[j] for j in idx]
        mean_distances = [mean_distances[j] for j in idx]
        final_files = [final_files[j] for j in idx]
        final_folders = [final_folders[j] for j in idx]
        final_features = [final_features[j] for j in idx]

        folder_list = [folder_list[j] for j in idx]
        feature_list = [feature_list[j] for j in idx]


    pca_folders = [*["me"]*len(input_images)]
    features = []
    for j in range(N_PCA):
        if j < len(folder_list):
            folder = folder_list[j]
            pca_folders.extend([f"Close example: {folder}"]*len(feature_list[j]))
            features.append(feature_list[j])
    for _ in range(5):#N_PCA):
        idx = np.random.randint(low=N_PCA, high=len(folder_list))
        folder = folder_list[idx]
        pca_folders.extend([f"Random example: {folder}"]*len(feature_list[idx]))
        features.append(feature_list[idx])
    features = np.array(features)
    features = features.reshape(features.shape[0]*features.shape[1], features.shape[-1])
    pca_features = [*own_features, *features]

    return final_files, final_folders, mean_distances, pca_features, pca_folders

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

        files, folders, distances, feature_list, current_folders = process_recorded_images(self.image_buffer)
        self.stop_loading_spinner()

        layout = QVBoxLayout()

        if files is None:
            label = QLabel("No face has been detected")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label)

        else:
            image = self.image_buffer[0]
            image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            pixmaps = [pixmap]
            for i in range(N_MATCHES):
                if i < len(files):
                    pixmaps.append(QPixmap(os.path.join(SUPPORT_SET_PATH, folders[i], files[i])))

            for i, pixmap in enumerate(pixmaps):
                if i >= 1:
                    label = QLabel(f"{folders[i-1]} {get_ranking(distances[i-1])}")
                    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    layout.addWidget(label)

                image_label = QLabel()
                image_label.setPixmap(pixmap.scaled(image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
                image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(image_label)

            image = get_PCA([*feature_list], current_folders)
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