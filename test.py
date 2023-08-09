from icrawler.builtin import BingImageCrawler

google_crawler = BingImageCrawler(storage={'root_dir': 'your_image_dir'})
google_crawler.crawl(keyword='cat', max_num=100)