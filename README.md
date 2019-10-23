# Israel-Sentiment-Pipline
The model will determine whether the Tweet’s is relevant to Israel domain, and predict it's sentiment toward Israel

## Overview
The model predicts the relevanve to the domain (Relevant/Irrelevant) and classify the sentiment toward Israel (-1,0,1).

* Relevance algorithm: Transformation of Tweets’ Sentiment in Political Communication; The Inverted Opinion Phenomenon, Supplement 2, page: 32
* Political leaning algorithm: Transformation of Tweets’ Sentiment in Political Communication; The Inverted Opinion Phenomenon, Supplement 3, page: 37

This github repository stores models code (model pipline) developed by Yogev Matalon & Ofir Magdaci, MSc Candidates @ Tel Aviv University. 

## Published Results
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3465153

## How to use
* Verify you're using python 3
* Install relevant packages (pandas, matplotlib, sklearn, XGboost)

### Prediction with the given model
* Open through jupyter notebook online_pipline.ipynb
* Run code

### Calibration
* Update entities, events dict and hashtags to your own needs
* Replace labeled_data.csv and test.csv files with your trained data
* Make sure you have emotion column (Watson tone analyzer extraction by IBM) - if not, set the emotion to False in configuration
* If you don't have users metadata set the user fetures to False in configuration
* Update configuration to your needs and run support-pipline.ipynb 

## Contact
Laboratory for Epidemic, Modeling and Analysis (LEMA), Tel Aviv University, Israel
http://danyamin.com 
