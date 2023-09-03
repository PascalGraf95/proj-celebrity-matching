import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.models import Sequential


MODEL_NAME = "best_encoder_505000_step_acc_0_9013.h5"#"best_encoder_419000_step_ap_0_2872_an_1_8216.h5"
SUPPORT_SET_PATH = "./Supportset_Celebrities"
DATABASE = "./database_celebrities_crop.json"

def read_image(path):
    image = tf.keras.preprocessing.image.load_img(path,
                                                  color_mode="rgb",
                                                  target_size=(300, 300),
                                                  interpolation="bilinear")
    image = tf.keras.preprocessing.image.img_to_array(image, dtype='float32')
    return np.expand_dims(image, axis=0)


def make_gradcam_heatmap(image, model):
    custom_submodel_1 = keras.models.Model(
        inputs=model.get_layer("xception").input, outputs=model.get_layer("xception").get_layer("block14_sepconv2_act").output
    )
    custom_submodel_2 = keras.models.Model(
        inputs=model.get_layer("xception").get_layer("global_average_pooling2d").input,
        outputs=model.get_layer("xception").get_layer("global_average_pooling2d").output)
    custom_submodel_3 = keras.models.Model(
        inputs=model.layers[1].input,
        outputs=model.output)

    # image_difference = model(img1) - model(img2)
    # sorted_indices = np.squeeze(np.argsort(image_difference))
    # print(image_difference[:, sorted_indices])

    with tf.GradientTape() as tape:
        conv_output = custom_submodel_1(image)
        pool = custom_submodel_2(conv_output)
        features = custom_submodel_3(pool)
        class_channel = features
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = conv_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = (tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap))
    # plt.matshow(heatmap)
    # plt.show()
    return heatmap.numpy()


def heatmap_overlay(heatmap, image):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    alpha = 0.4
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + image
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    return superimposed_img



if __name__ == '__main__':
    encoder = tf.keras.models.load_model(os.path.join('vgg_models', MODEL_NAME), compile=False)
    encoder.summary(expand_nested=True)

    image1 = read_image(r"A:\Arbeit\Github\proj-celebrity-matching\Supportset_Celebrities_crop\Andre Agassi\000011.jpg")
    image2 = read_image(r"A:\Arbeit\Github\proj-celebrity-matching\Supportset_Celebrities_crop\Andre Agassi\000012.jpg")

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(image1, encoder)
    overlay_image = heatmap_overlay(heatmap, image1[0])

    plt.imshow(overlay_image)
    plt.show()

    heatmap = make_gradcam_heatmap(image2, encoder)
    overlay_image = heatmap_overlay(heatmap, image2[0])

    plt.imshow(overlay_image)
    plt.show()

    # Display heatmap



