import face_recognition
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import argparse

INPUT_PATH = "./Supportset_Celebrities"
OUTPUT_PATH = "./Supportset_Celebrities_crop"
image_extensions = [".png", ".jpg", '.jpeg']


def main(delete_faulty_images, min_image_size):
    # Get all celebrity image folders and iterate through them
    for folder in tqdm([folder for folder in os.listdir(INPUT_PATH) if os.path.isdir(os.path.join(INPUT_PATH,folder))]):
        for image_path in os.listdir(os.path.join(INPUT_PATH, folder)):
            # If not an image, delete the file.
            if not any(ext in image_path for ext in image_extensions):
                os.remove(os.path.join(INPUT_PATH, image_path))
                print(f"Deleted non-image file: {image_path}")

            # Don't redo existing images that have been cropped already
            input_path = os.path.join(INPUT_PATH, folder, image_path)
            output_image_path = os.path.join(OUTPUT_PATH, folder, image_path)
            if not os.path.exists(output_image_path):
                os.makedirs(os.path.join(OUTPUT_PATH, folder), exist_ok=True)

                # Check for faces in image and skip if necessary
                img = face_recognition.load_image_file(input_path)
                face_locations = face_recognition.face_locations(img)

                if len(face_locations) == 0:
                    print(f"No face detected in: {input_path}")
                    if delete_faulty_images:
                        os.remove(input_path)
                    continue

                # Open image as array
                image = np.array(Image.open(input_path))

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
                if any(dim < min_image_size for dim in cropped_image.shape[:2]):
                    print(f"Face detected in {input_path} is too small in resolution.")
                    if delete_faulty_images:
                        os.remove(input_path)
                    continue

                im_pil = Image.fromarray(cropped_image)
                if im_pil.mode in ("RGBA", "P"):
                    im_pil = im_pil.convert("RGB")
                im_pil.save(output_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and optionally delete faulty images.")
    parser.add_argument('-del', '--delete_faulty_images', type=bool, nargs="?", default=False,
                        help="Delete faulty images (True or False)", required=False)
    parser.add_argument('-min', '--min_image_size', type=bool, default=150, required=False)

    args = parser.parse_args()
    main(args.delete_faulty_images, args.min_image_size)
