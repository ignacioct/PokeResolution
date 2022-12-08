"""
SuperRes Visualization app

Streamlit App that allows to visualize how the different SuperRes models
works on a preset of test images.
"""

import numpy as np
import streamlit as st
from tensorflow import clip_by_value, expand_dims, squeeze, split, concat
from tensorflow.image import resize, ResizeMethod, rgb_to_yuv, yuv_to_rgb, resize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.engine.functional import Functional
from tensorflow.random import set_seed


def get_test_subset(path: str) -> dict:
    """
    Obtains a subset of 10 pokemon to showcase on the app and allow
    inference.

    Parameters
    ----------
    path: str
        path to the test dataset.

    Returns
    -------
    img_list: [str: np.ndarray]
        list with images from the test dataset.
    """

    # Constants and variables
    SUBSET_LEN = 10
    img_list = []
    index = 0

    # Loading dataset
    test_dataset = image_dataset_from_directory(path, image_size=(256, 256), seed=123)

    # Obtain 10 first images
    for element in test_dataset.as_numpy_iterator():
        for img in element[0]:
            index += 1
            img_list.append(img)
            if index >= SUBSET_LEN:
                break

    return img_list


def predict(input_image: np.array, model: Functional) -> np.ndarray:
    """
    Given an input image, predict the output.

    Parameters
    ----------
    input_image: np.array
        image that is feeded as input to the model
    model:
        model to feed the image and create the prediction that has be chosen
        via UI.

    Returns
    -------
    output_image: np.array
        image upscaled by the model.
    """

    return model(input_image)


def downscale_image(chosen_img: np.ndarray, input_size: int) -> np.ndarray:
    """
    Given an input image, downscale it to the given factor.
    Downscaling is performed using the bicubic interpolation method.

    Parameters
    ----------
    chosen_img: np.ndarray
        image to downscale.
    input_size: int
        size to downscale the image to.

    Returns
    -------
    downscaled_img: np.ndarray
        downscaled imaged.
    """

    return clip_by_value(
        resize(
            chosen_img,
            [input_size, input_size],
            method=ResizeMethod.BICUBIC,
            antialias=True,
        ),
        0,
        255,
    )


def load_chosen_model(models_dict, chosen_model_str, downscale):
    """
    Given the chosen parameters through the UI, and the dictionary with the different
    model names and paths, returns the chosen model.
    """

    list_index = 0

    if downscale == 2:
        list_index = 0
    elif downscale == 4:
        list_index = 1

    selected_model = models_dict[chosen_model_str][list_index]

    rgb = True
    if chosen_model_str != "EDSR" and chosen_model_str != "bestEDSR":
        rgb = False

    return load_model(selected_model, compile=False), rgb


def processs_input(input):
    input = rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = split(input, 3, axis=last_dimension_axis)
    return y, u, v


def main():

    st.set_page_config("Poke Resolution Showdown", "🦊")

    # Constants
    MODEL_LIST = ["EDSR", "ourSRCNN", "SRCNN", "bestEDSR"]

    FACTOR_LIST = ["Factor 2", "Factor 4"]

    TEST_PATH = "images/test/"

    MODELS_DICT = {
        "EDSR": [
            "saved/model-edsr-x2-a10-c3-e15-mean_absolute_error.h5",
            "saved/model-edsr-x4-a10-c3-e15-mean_absolute_error.h5",
        ],
        "ourSRCNN": [
            "saved/model-oursrcnn-x2-a10-c1-e15-mean_absolute_error.h5",
            "saved/model-oursrcnn-x4-a10-c1-e15-mean_absolute_error.h5",
        ],
        "SRCNN": [
            "saved/model-srcnn-x2-a10-c1-e15-mean_absolute_error.h5",
            "saved/model-srcnn-x4-a10-c1-e15-mean_absolute_error.h5",
        ],
        "bestEDSR": [
            "saved/model-edsr-x2-a10-c3-e40-mean_squared_error-rs0.1.h5",
            "saved/model-edsr-x4-a10-c3-e40-mean_squared_error-rs0.1.h5",
        ],
    }

    # Set random seed
    set_seed(123)

    # Dataset
    subset_test = get_test_subset(TEST_PATH)

    st.title("Poke Resolution Showdown App")
    st.subheader("An app made by Ignacio, Rafa & Juan")

    st.write(
        "In this app you can try the different super resolution models that we have developed on a subset of Pokemon images from the test dataset. The models have never seen these images."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.header("Pick a model and a factor")
        chosen_model = st.selectbox(options=MODEL_LIST, label="")
        chosen_factor = st.radio(options=FACTOR_LIST, label="")

        if chosen_factor == "Factor 2":
            downscale_factor = 2
            input_size = 128
        elif chosen_factor == "Factor 4":
            downscale_factor = 4
            input_size = 64

    with col2:
        st.header("Pick a pokemon from the test dataset")
        selected_pokemon = st.slider(min_value=1, max_value=10, label="")

    chosen_image = subset_test[selected_pokemon]
    downscaled_image = downscale_image(chosen_image, input_size)

    # Loading the model
    model, rgb = load_chosen_model(MODELS_DICT, chosen_model, chosen_factor)

    lr_input = ""

    # Predict image
    if not rgb:
        y, cb, cr = processs_input(downscaled_image)
        lr_input = y
    else:
        lr_input = downscaled_image

    predicted_image = predict(expand_dims(lr_input, axis=0), model)[0]

    if not rgb:
        sr = predicted_image
        out_img_cb = resize(cb, [sr.shape[0], sr.shape[0]], method=ResizeMethod.BICUBIC)
        out_img_cr = resize(cr, [sr.shape[0], sr.shape[0]], method=ResizeMethod.BICUBIC)
        sr = concat([sr, out_img_cb, out_img_cr], axis=-1)
        sr = clip_by_value(yuv_to_rgb(sr), 0, 255)
        predicted_image = sr

    col1, col2 = st.columns(2)

    with col1:
        st.header("Original image")
        st.image(
            array_to_img(downscaled_image),
            width=264,
        )

    with col2:
        st.header("After super resolution")
        st.image(
            array_to_img(predicted_image),
            width=264,
        )


if __name__ == "__main__":
    main()
