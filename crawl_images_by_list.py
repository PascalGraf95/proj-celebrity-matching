from icrawler.builtin import BingImageCrawler
import os

# List of names for which you want to download images
names = [
    "Abby Warmbach",
    "Abigail Breslin",
    "Adam Peaty",
    "Adel Tawil",
]

# Number of images to download for each person
num_images = 20

# Specify the output directory where images will be saved
root_dir = "downloaded_images"

# Create the output directory if it doesn't exist


search_filters = dict(
    size="medium",
    type="photo",
    color="color"
)

# Loop through each name and download images
for name in names:
    output_directory = os.path.join(root_dir, name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Initialize the ImageCrawler
    google_crawler = BingImageCrawler(storage={"root_dir": output_directory},
                                      feeder_threads=1,
                                      parser_threads=1,
                                      downloader_threads=6)
    google_crawler.crawl(keyword=name, max_num=num_images, filters=search_filters)

print("Image download complete.")