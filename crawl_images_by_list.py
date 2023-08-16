from icrawler.builtin import BingImageCrawler
import os

# List of names for which you want to download images
names = [
    "Robert De Niro",
    "Meryl Streep",
    "Leonardo DiCaprio",
    "Tom Hanks",
    "Jennifer Lawrence",
    "Denzel Washington",
    "Cate Blanchett",
    "Johnny Depp",
    "Scarlett Johansson",
    "Brad Pitt",
    "Julia Roberts",
    "Morgan Freeman",
    "Charlize Theron",
    "Harrison Ford",
    "Nicole Kidman",
    "Samuel L. Jackson",
    "Emma Stone",
    "Anthony Hopkins",
    "Natalie Portman",
    "Al Pacino",
    "Angelina Jolie",
    "Will Smith",
    "Viola Davis",
    "Daniel Day-Lewis",
    "Sandra Bullock",
    "Ryan Gosling",
    "Gong Li",
    "Hugh Jackman",
    "Keanu Reeves",
    "Marion Cotillard"
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