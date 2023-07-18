# importing the necessary modules:
import numpy as np
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
	balloon_mask = np.array(Image.open("img_dir/balloons.png"))

	image_colors = ImageColorGenerator(balloon_mask)

	wc_balloons = WordCloud(stopwords=STOPWORDS,
							background_color="white",
							mode="RGBA",
							max_words=1000,
							# contour_width=3,
							repeat=True,
							mask=balloon_mask)

	text = open("data/birthday_text.txt").read()
	wc_balloons.generate(text)
	wc_balloons.recolor(color_func=image_colors)

	plt.imshow(wc_balloons)
	plt.axis("off")
	plt.show()