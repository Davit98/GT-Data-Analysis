# GT-Data-Analysis

This repo includes all the code used for analyzing the collected google search history data. A short description for each file is presented below:

Main.ipyb - this jupyter notebook shows the main data analysis steps and reproduces all results given in the report. If Main.ipynb is not rendering on github, you can view it [here](https://nbviewer.jupyter.org/github/Davit98/GT-Data-Analysis/blob/main/Main.ipynb).

annotations.docx - this is a microsoft word document containing the 167 annotated queries from Sample#6.

categories.txt - contains the 8 predefined categories and their corresponding context words. Each row is one line and is of the format CATEGORY\CONTEXT_WORD.

data_processing.py - contains functions for doing textual data preprocessing (e.g. tokenization, lemmatization).

data_viz.py - contains code for producing insightful visualizations.

empath_helper.py - helper functions for using Empath.

query_categorization.py - contains both the original and modified version of the algorithm used for doing web search query categorization 

utils.py - utility functions such as reading the categories from categories.txt file.

