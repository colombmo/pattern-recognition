# How to run

From keywordSpotting directory, execute:

- python cutImages.py : To extract the single word images from the full pages, normalize and binarize them.
- python extractFeatures.py: To extract some features from the images, and normalize them. This is saved in some text file, in such a way that during recognition the images don't have to be analysed again.
- python recognition.py: To iterate through all keywords defined in #### keywords.txt #### and find them inside the test images. 
- The results of the recognition will be stored in the file #### results.txt ####

# Parameters and features selection

During validation, we tried several combinations of parameters and features, and we found out that we were getting the best results with the following features:
- Upper contour
- Lower contour
- Number of black-white transitions
- Original ratio width/height of each word

We used DTW with a window width of 1px, on images of the words after having been scaled to 50px x 100px.

With those parameters and features, we managed to get 54.5% mean average precision on the validation set.

# Description of data

##Task ##
Your task is to develop a machine learning approach for spotting keywords in the provided documents.
You can test your approach on the provided training and validation dataset where you find a list of keywords that you can find for certain at least once in each set.


## Data ##
In this repository you'll find all the data necessary for your KeywordSpotting Task.

You find the following folders:


### ground-truth ###
Contains ground-truth data.

#### transcription.txt ####

Contains the transcription of all words (on a character level) of the whole dataset. The Format is as follows:

	- XXX-YY-ZZ: XXX = Document Number, YY = Line Number, ZZ = Word Number
	- Contains the character-wise transcription of the word (letters seperated with dashes)
	- Special characters denoted with s_
		- numbers (s_x)
		- punctuation (s_pt, s_cm, ...)
		- strong s (s_s)
		- hyphen (s_mi)
		- semicolon (s_sq)
		- apostrophe (s_qt)
		- colon (s_qo)

#### locations #####

Contains bounding boxes for all words in the svg-format.

	- XXX.svg: File containing the bounding boxes for the given documents
	- **id** contains the same XXX-YY-ZZ naming as above

### images ###

Contains the original images in jpg-format.

### task ###
Contains three files:

####train.txt / valid.txt ####
Contains a splitting of the documents into a training and a validation set.


#### keywords.txt ####
Contains a list of keywords of which each will be at least **once** in the training and validation dataset.
