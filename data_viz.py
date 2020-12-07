import numpy as np
import matplotlib.pyplot as plt

from utils import get_human_readable_quarter_name

from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image


def circle_time_series_plot(top_words,
							n,
							colors, 
							title_fontsize=40, 
							text_fontsize=22, 
							xlabel_fontsize=25,
							ylabel_fontsize=25,
							fig_w=40,
							fig_h=6,
							text_rotation='20',
							r=2.2):
	"""
	Produces a plot showing the top-n most frequent words/phrases searched per quarter. 
	Each word is represnted with a circle, where bigger circle means more searches. 

	Parameters
	----------
	top_words : dict
		Dictionary containing top-n words with corresponding frequences per each quarter. 

	n : int
		Number of top-n words to consider. 

	colors : list
		List of colours for the circles. Should contain number of colours equal to n.  

	title_fontsize : int, optional
		Font size of the plot's title.

	text_fontsize : int, optional
		Font size of the circles' labels.

	xlabel_fontsize : int, optional
		Font size of the x-axis labels.

	ylabel_fontsize : int, optional
		Font size of the y-axis labels.

	fig_w : int, optional
		Figure width.

	fig_h : int, optional
		Figure height.

	text_rotation : str, optional
		Rotation angle of circles' labels. 

	r : int or float, optional
		Coefficient to control the size of circles. Higher coefficient creates smaller circles. r must be greater than or equal to 1.
	"""

	xaxis = list(top_words.keys())
	xaxis_ticks = [get_human_readable_quarter_name(e) for e in xaxis]

	mx = max([e[1] for _,v in top_words.items() for e in v])

	figure, ax = plt.subplots(figsize=(fig_w,fig_h))

	ax.set_xlim((0,len(xaxis)+1))
	ax.set_ylim((0,n+1))

	for i, e in enumerate(top_words.values()):
		j = n
		for w in e:
			circle = plt.Circle((i+1,j),w[1]/(r*mx),color=colors[j-1])
			ax.add_artist(circle)
			ax.text(i + 1, j, w[0], fontsize=text_fontsize, color='black', rotation=text_rotation)
			j-=1

	top_n_labels = ['Top 3rd word', 'Top 2nd word', 'Top 1st word']

	for k in range(4,n+1):
		top_n_labels.insert(0,'Top {}th word'.format(k))

	plt.title('Top-{} most frequent words/phrases searched per quarter'.format(n),fontsize=title_fontsize)    
	plt.xticks(list(range(1,len(xaxis_ticks)+1)),xaxis_ticks,rotation='45',fontsize=xlabel_fontsize);
	plt.yticks(list(range(1,n+1)),top_n_labels,fontsize=ylabel_fontsize);




def search_rate_dynamics_plot(history):
	"""
	Produces a plot showing the search rate dynamics of the top-n most frequent words/phrases.

	Parameters
	----------
	history : dict
		Dictionary containing top-n words with corresponding frequences per each quarter. 

	"""
	xaxis = list(history[next(iter(history))].keys())
	xaxis_ticks = [get_human_readable_quarter_name(e) for e in xaxis]

	figure, ax = plt.subplots(figsize=(40,20))
	ax.set_xlim((0,len(xaxis)-1))

	for w in history:
		ax.plot(list(range(len(xaxis))),list(history[w].values()),linewidth=5)

	plt.ylabel('Search frequency', fontsize=30)
	plt.title('Top-{} most frequent words\'/phrases\' search rate dynamics'.format(len(history)),fontsize=40)    
	plt.legend(list(history.keys()),fontsize=30)
	plt.xticks(list(range(len(xaxis_ticks))),xaxis_ticks,rotation='45',fontsize=30);
	plt.yticks(fontsize=30);	




def word_cloud_plot(docs,background_image_path):
	""" 
	Produces a word cloud plot based on the input data with a given background image.

	Parameters
	----------
	docs : list
		List of tokenized strings.

	background_image_path : str
		Path of the background image.
	"""
	text = " ".join([word for doc in docs for word in doc])
	mask = np.array(Image.open(background_image_path))
	wordcloud_fig = WordCloud(background_color="white", max_words=1000, mask=mask).generate(text)

	# create coloring from image
	image_colors = ImageColorGenerator(mask)
	plt.figure(figsize=[50,50])
	plt.imshow(wordcloud_fig.recolor(color_func=image_colors), interpolation="bilinear")
	plt.axis("off")






