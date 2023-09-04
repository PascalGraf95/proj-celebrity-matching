from icrawler.builtin import BingImageCrawler
import os

# List of names for which you want to download images
names = [
    "Barack Obama", "Angela Merkel", "Vladimir Putin", "Xi Jinping", "Narendra Modi",
    "Emmanuel Macron", "Justin Trudeau", "Theresa May", "Boris Johnson", "Joe Biden",
    "Hassan Rouhani", "Benjamin Netanyahu", "Shinzo Abe", "Jacinda Ardern", "Joko Widodo",
    "Kim Jong-un", "Recep Tayyip Erdoğan", "Andrés Manuel López Obrador", "Luis Abinader",
    "Mário Draghi", "Abiy Ahmed", "Volodymyr Zelensky", "Kamala Harris", "Michelle Bachelet",
    "Aung San Suu Kyi", "Nicolas Maduro", "Tsai Ing-wen", "Imran Khan", "Mahathir Mohamad",
    "Cyril Ramaphosa", "Felipe Calderón", "Julia Gillard", "Dilma Rousseff", "Cristina Fernández de Kirchner",
    "Rafael Correa", "Juan Manuel Santos", "Álvaro Uribe", "Luiz Inácio Lula da Silva", "Vicente Fox",
    "George W. Bush", "Bill Clinton", "Tony Blair", "Jacques Chirac", "Gerhard Schröder",
    "Helmut Kohl", "Margaret Thatcher", "Nelson Mandela", "Mikhail Gorbachev", "Boutros Boutros-Ghali",
    "Lech Wałęsa", "Brian Mulroney", "Yitzhak Rabin", "Kim Campbell", "Hosni Mubarak",
    "Fidel Castro", "Bashar al-Assad", "Saddam Hussein", "Mohammad Khatami", "Slobodan Milošević",
    "Nawaz Sharif", "Sheikh Hasina", "Benazir Bhutto", "Abdullah Gül", "Ehud Barak",
    "Shimon Peres", "Ariel Sharon", "Ehud Olmert", "Yasser Arafat", "Ayatollah Ali Khamenei",
    "Nawaz Sharif", "Aung San Suu Kyi", "Khaleda Zia", "Imran Khan", "Thaksin Shinawatra",
    "Anwar Ibrahim", "Mahathir Mohamad", "Lee Hsien Loong", "Kim Dae-jung", "Park Geun-hye",
    "Shinzo Abe", "Shinzo Abe", "Narendra Modi", "Atal Bihari Vajpayee", "Rajiv Gandhi",
    "Benazir Bhutto", "Pervez Musharraf", "Nawaz Sharif", "Sheikh Hasina", "Khaleda Zia",
    "Emmerson Mnangagwa", "Robert Mugabe", "Nelson Chamisa", "Julius Malema", "Cyril Ramaphosa",
    "Jacob Zuma", "Olusegun Obasanjo", "Goodluck Jonathan", "Muhammadu Buhari", "Paul Biya",
    "Yoweri Museveni", "Robert Mugabe", "Morgan Tsvangirai", "Paul Kagame", "Raila Odinga",
    "Uhuru Kenyatta", "Abiy Ahmed", "Hailemariam Desalegn", "Alassane Ouattara", "Laurent Gbagbo",
    "Nana Akufo-Addo", "Jerry Rawlings", "Muammar Gaddafi", "Fayez al-Sarraj", "Abdul Hamid Dbeibeh",
    "Idriss Déby", "Félix Tshisekedi", "Joseph Kabila", "Paul Kagame", "Yoweri Museveni"
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