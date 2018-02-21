---
title: Open Datasets for Deep Learning
layout: default
---

# Open Data for Deep Learning

Here you'll find an organized list of interesting, high-quality datasets for machine learning research. We welcome your contributions for [curating this list](https://github.com/deeplearning4j/deeplearning4j/blob/gh-pages/opendata.md)! You can find other lists of such datasets [on Wikipedia](https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research), for example.

## Recent Additions

* [Open Source Biometric Recognition Data](http://openbiometrics.org/)
* [Google Audioset](https://research.google.com/audioset/): An expanding ontology of 632 audio event classes and a collection of 2,084,320 human-labeled 10-second sound clips drawn from YouTube videos.
* [Uber 2B trip data](https://movement.uber.com/cities): Slow rollout of access to ride data for 2Bn trips.
* [Yelp Open Dataset](https://www.yelp.com/dataset): The Yelp dataset is a subset of Yelp businesses, reviews, and user data for use in NLP.
* [Core50: A new Dataset and Benchmark for Continuous Object Recognition](https://vlomonaco.github.io/core50/)
* [Kaggle Datasets Page](https://www.kaggle.com/datasets)
* [Data Portals](http://dataportals.org/)
* [Open Data Monitor](https://opendatamonitor.eu/)
* [Quandl Data Portal](https://www.quandl.com/)

<p align="center">
<a href="https://docs.skymind.ai/docs/welcome" type="button" class="btn btn-lg btn-success" onClick="ga('send', 'event', ‘quickstart', 'click');">GET STARTED WITH DEEP LEARNING</a>
</p>

## Natural-Image Datasets

* [MNIST: handwritten digits](http://yann.lecun.com/exdb/mnist/): The most commonly used sanity check. Dataset of 25x25, centered, B&W handwritten digits. It is an easy task — just because something works on MNIST, doesn’t mean it works.
* [CIFAR10 / CIFAR100]( http://www.cs.utoronto.ca/~kriz/cifar.html): 32x32 color images with 10 / 100 categories. Not commonly used anymore, though once again, can be an interesting sanity check.
* [Caltech 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/): Pictures of objects belonging to 101 categories.
* [Caltech 256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/): Pictures of objects belonging to 256 categories.
* [STL-10 dataset](http://cs.stanford.edu/~acoates/stl10/): is an image recognition dataset for developing unsupervised feature learning, deep learning, self-taught learning algorithms. Like CIFAR-10 with some modifications.
* [The Street View House Numbers (SVHN)](http://ufldl.stanford.edu/housenumbers/): House numbers from Google Street View. Think of this as recurrent MNIST in the wild.
* [NORB](http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/): Binocular images of toy figurines under various illumination and pose.
* [Pascal VOC](http://pascallin.ecs.soton.ac.uk/challenges/VOC/): Generic image Segmentation / classification — not terribly useful for building real-world image annotation, but great for baselines
* [Labelme](http://labelme.csail.mit.edu/Release3.0/browserTools/php/dataset.php): A large dataset of annotated images.
* [ImageNet](http://image-net.org/): The de-facto image dataset for new algorithms. Many image API companies have labels from their REST interfaces that are suspiciously close to the 1000 category; WordNet; hierarchy from ImageNet.
* [LSUN](http://lsun.cs.princeton.edu/2016/): Scene understanding with many ancillary tasks (room layout estimation, saliency prediction, etc.) and an associated competition.
* [MS COCO](http://mscoco.org/): Generic image understanding / captioning, with an associated competition.
* [COIL 20](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php): Different objects imaged at every angle in a 360 rotation.
* [COIL100 ](http://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php): Different objects imaged at every angle in a 360 rotation.
* [Google's Open Images](https://research.googleblog.com/2016/09/introducing-open-images-dataset.html): A collection of 9 million URLs to images "that have been annotated with labels spanning over 6,000 categories" under Creative Commons.

### Geospatial data

* [OpenStreetMap](http://wiki.openstreetmap.org/wiki/Planet.osm): Vector data for the entire planet under a free license. It contains (an older version of) the US Census Bureau’s data.
* [Landsat8](https://landsat.usgs.gov/landsat-8): Satellite shots of the entire Earth surface,  updated every several weeks.
* [NEXRAD](https://www.ncdc.noaa.gov/data-access/radar-data/nexrad):  Doppler radar scans of atmospheric conditions in the US.

## Artificial Datasets

* [Arcade Universe](https://github.com/caglar/Arcade-Universe): - An artificial dataset generator with images containing arcade games sprites such as tetris pentomino/tetromino objects. This generator is based on the O. Breleux’s bugland dataset generator.
* A collection of datasets inspired by the ideas from [BabyAISchool](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/BabyAISchool)
* [BabyAIShapesDatasets](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/BabyAIShapesDatasets): distinguishing between 3 simple shapes
* [BabyAIImageAndQuestionDatasets](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/BabyAIImageAndQuestionDatasets): a question-image-answer dataset
* Datasets generated for the purpose of an empirical evaluation of deep architectures ([DeepVsShallowComparisonICML2007](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/DeepVsShallowComparisonICML2007)):
* [MnistVariations](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations): introducing controlled variations in MNIST
* [RectanglesData](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/RectanglesData): discriminating between wide and tall rectangles
* [ConvexNonConvex](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/ConvexNonConvex): discriminating between convex and nonconvex shapes
* [BackgroundCorrelation](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/BackgroundCorrelation): controling the degree of correlation in noisy MNIST backgrounds.

## Facial Datasets

* [Labelled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/): 13,000 cropped facial regions (using; [Viola-Jones](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework) that have been labeled with a name identifier. A subset of the people present have two images in the dataset — it’s quite common for people to train facial matching systems here.
* [UMD Faces](http://www.umdfaces.io) Annotated dataset of 367,920 faces of 8,501 subjects.
* [CASIA WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) Facial dataset of 453,453 images over 10,575 identities after face detection. Requires some filtering for quality.
* [MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/) 1 million images of celebrities from around the world. Requires some filtering for best results on deep networks.
* [Olivetti](http://www.cs.nyu.edu/~roweis/data.html): A few images of several different people.
* [Multi-Pie](http://www.multipie.org/): The CMU Multi-PIE Face Database
* [Face-in-Action](http://www.flintbox.com/public/project/5486/)
* [JACFEE](http://www.humintell.com/jacfee/): Japanese and Caucasian Facial Expressions of Emotion
* [FERET](http://www.itl.nist.gov/iad/humanid/feret/feret_master.html): The Facial Recognition Technology Database
* [mmifacedb](http://www.mmifacedb.com/): MMI Facial Expression Database
* [IndianFaceDatabase](http://vis-www.cs.umass.edu/~vidit/IndianFaceDatabase/)
* [The Yale Face Database](http://vision.ucsd.edu/content/yale-face-database) and [The Yale Face Database B](http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html)).

## Video Datasets

* [Youtube-8M](https://research.googleblog.com/2016/09/announcing-youtube-8m-large-and-diverse.html): A large and diverse labeled video dataset for video understanding research.


## Text Datasets

* [20 newsgroups](http://qwone.com/~jason/20Newsgroups/): Classification task, mapping word occurences to newsgroup ID. One of the classic datasets for text classification) usually useful as a benchmark for either pure classification or as a validation of any IR / indexing algorithm.
* [Reuters News dataset](https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection): (Older) purely classification-based dataset with text from the newswire. Commonly used in tutorial.
* [Penn Treebank](http://www.cis.upenn.edu/~treebank/): Used for next word prediction or next character prediction.
* [UCI’s Spambase](https://archive.ics.uci.edu/ml/datasets/Spambase): (Older) classic spam email dataset from the famous UCI Machine Learning Repository. Due to details of how the dataset was curated, this can be an interesting baseline for learning personalized spam filtering.
* [Broadcast News](http://www.ldc.upenn.edu/Catalog/CatalogEntry.jsp?catalogId=LDC97S44): Large text dataset, classically used for next word prediction.
* [Text Classification Datasets](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M):  From; Zhang et al., 2015;  An extensive set of eight datasets for text classification. These are the benchmark for new text classification baselines. Sample size of 120K to 3.6M, ranging from binary to 14 class problems. Datasets from DBPedia, Amazon, Yelp, Yahoo! and AG.
* [WikiText](http://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/): A large language modeling corpus from quality Wikipedia articles, curated by Salesforce MetaMind.
* [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/): The Stanford Question Answering Dataset — broadly useful question answering and reading comprehension dataset, where every answer to a question is posed as a segment of text.
* [Billion Words dataset](http://www.statmt.org/lm-benchmark/): A large general-purpose language modeling dataset. Often used to train distributed word representations such as word2vec.
* [Common Crawl](http://commoncrawl.org/the-data/):  Petabyte-scale crawl of the web — most frequently used for learning word embeddings. Available for free from Amazon S3. Can also be useful as a network dataset for it’s a crawl of the WWW.
* [Google Books Ngrams](https://aws.amazon.com/datasets/google-books-ngrams/): Successive words from Google books. Offers a simple method to explore when a word first entered wide usage.
* [Yelp Open Dataset](https://www.yelp.com/dataset): The Yelp dataset is a subset of Yelp businesses, reviews, and user data for use in NLP.

### Question answering

* [Maluuba News QA Dataset](https://datasets.maluuba.com/NewsQA): 120K Q&A pairs on CNN news articles.
* [Quora Question Pairs](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs): first dataset release from Quora containing duplicate / semantic similarity labels.
* [CMU Q/A Dataset](http://www.cs.cmu.edu/~ark/QA-data/): Manually-generated factoid question/answer pairs with difficulty ratings from Wikipedia articles.
* [Maluuba goal-oriented dialogue](https://datasets.maluuba.com/Frames): Procedural conversational dataset where the dialogue aims at accomplishing a task or taking a decision. Often used to work on chat bots.
* [bAbi](https://research.fb.com/projects/babi/): Synthetic reading comprehension and question answering datasets from Facebook AI Research (FAIR).
* [The Children’s Book Test](http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz): Baseline of (Question + context, Answer) pairs extracted from Children’s books available through Project Gutenberg. Useful for question-answering (reading comprehension)  and factoid look-up.

### Sentiment

* [Multidomain sentiment analysis dataset](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/) An older, academic dataset.
* [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/): An older, relatively small dataset for binary sentiment classification. Fallen out of favor for benchmarks in the literature in lieu of larger datasets.
* [Stanford Sentiment Treebank](http://nlp.stanford.edu/sentiment/code.html): Standard sentiment dataset with fine-grained sentiment annotations at every node of each sentence’s parse tree.

## Recommendation and ranking systems

* [Movielens](https://grouplens.org/datasets/movielens/): Movie ratings dataset from the Movielens website, in various sizes ranging from demo to mid-size.
* [Million Song Dataset](https://www.kaggle.com/c/msdchallenge): Large, metadata-rich, open source dataset on Kaggle that can be good for people experimenting with hybrid recommendation systems.
* [Last.fm](http://grouplens.org/datasets/hetrec-2011/): Music recommendation dataset with access to underlying social network and other metadata that can be useful for hybrid systems.
* [Book-Crossing dataset](http://www.informatik.uni-freiburg.de/~cziegler/BX/):: From the Book-Crossing community. Contains 278,858 users providing 1,149,780 ratings about 271,379 books.
* [Jester](http://www.ieor.berkeley.edu/~goldberg/jester-data/): 4.1 million continuous ratings (-10.00 to +10.00) of 100 jokes from 73,421 users.
* [Netflix Prize](http://www.netflixprize.com/):: Netflix released an anonymized version of their movie rating dataset; it consists of 100 million ratings, done by 480,000 users who have rated between 1 and all of the 17,770 movies. First major Kaggle style data challenge. Only available unofficially, as privacy issues arose.

## Networks and Graphs
* [Amazon Co-Purchasing](http://snap.stanford.edu/data/#amazon): Amazon Reviews  crawled data from "the users who bought this also bought…” section of Amazon, as well as Amazon review data for related products. Good for experimenting with recommendation systems in networks.
* [Friendster Social Network Dataset](https://archive.org/details/friendster-dataset-201107): Before their pivot as a gaming website, Friendster released anonymized data in the form of friends lists for 103,750,348 users.

## Speech Datasets

* [2000 HUB5 English](https://catalog.ldc.upenn.edu/LDC2002T43):  English-only speech data used most recently in the Deep Speech paper from Baidu.
* [LibriSpeech](http://www.openslr.org/12/): Audio books data set of text and speech. Nearly 500 hours of clean speech of various audio books read by multiple speakers, organized by chapters of the book containing both the text and the speech.
* [VoxForge](http://www.voxforge.org/): Clean speech dataset of accented english. Useful for instances in which you expect to need robustness to different accents or intonations.
* [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1):  English-only speech recognition dataset.
* [CHIME](http://spandh.dcs.shef.ac.uk/chime_challenge/data.html): Noisy speech recognition challenge dataset. Dataset contains real simulated and clean voice recordings. Real being actual recordings of 4 speakers in nearly 9000 recordings over 4 noisy locations,  simulated is generated by combining multiple environments over speech utterances and clean being non-noisy recordings.
* [TED-LIUM](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus):  Audio transcription of TED talks. 1495 TED talks audio recordings along with full text transcriptions of those recordings.

## Symbolic Music Datasets

* [Piano-midi.de: classical piano pieces](http://www.piano-midi.de/)
* [Nottingham : over 1000 folk tunes](http://abc.sourceforge.net/NMD/)
* [MuseData: electronic library of classical music scores](http://musedata.stanford.edu/)
* [JSB Chorales: set of four-part harmonized chorales](http://www.jsbchorales.net/index.shtml)

## Miscellaneous Datasets

* [CMU Motion Capture Database](http://mocap.cs.cmu.edu/)
* [Brodatz dataset: texture modeling](http://www.ux.uis.no/~tranden/brodatz.html)
* [300 terabytes of high-quality data from the Large Hadron Collider (LHC) at CERN](http://opendata.cern.ch/search?ln=en&p=Run2011A+AND+collection%3ACMS-Primary-Datasets+OR+collection%3ACMS-Simulated-Datasets+OR+collection%3ACMS-Derived-Datasets)
* [NYC Taxi dataset](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml): NYC taxi data obtained as a result of a FOIA request, led to [privacy issues](https://research.neustar.biz/2014/09/15/riding-with-the-stars-passenger-privacy-in-the-nyc-taxicab-dataset/).
* [Uber FOIL dataset](https://github.com/fivethirtyeight/uber-tlc-foil-response): Data for 4.5M pickups in NYC from an Uber FOIL request.
* [Criteo click stream dataset](http://research.criteo.com/outreach/): Large Internet advertisement dataset from a major EU retargeter.

### Health & Biology Data

* [EU Surveillance Atlas of Infectious Diseases](http://ecdc.europa.eu/en/data-tools/atlas/Pages/atlas.aspx)
* [Merck Molecular Activity Challenge](http://www.kaggle.com/c/MerckActivity/data)
* [Musk dataset](https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2)): The Musk database describes molecules occurring in different conformations. Each molecule is either musk or non-musk and one of the conformations determines this property.

### Government & statistics data

* [Data USA: The most comprehensive visualization of US public data](http://datausa.io)
* [EU Gender statistics database](http://eige.europa.eu/gender-statistics)
* [The Netherlands' Nationaal Georegister](http://www.nationaalgeoregister.nl/geonetwork/srv/dut/search#fast=index&from=1&to=50&any_OR_geokeyword_OR_title_OR_keyword=landinrichting*&relation=within) (Dutch)
* [United Nations Development Programme Projects](http://open.undp.org/#2016)

## <a name="intro">Introductory Machine-Learning Resources</a>

For people just getting started with deep learning, the following tutorials and videos provide an easy entrance to the fundamental ideas of deep neural networks:

* [Beginner's Guide to Deep Neural Networks](./neuralnet-overview)
* [Deep Reinforcement Learning](./deepreinforcementlearning)
* [Deep Convolutional Networks (CNNs) for Images](./convolutionalnets)
* [Recurrent Networks and LSTMs](./lstm)
* [Generative Adversarial Networks (GAN)](./generative-adversarial-network)
* [Multilayer Perceptron (MLP) for Classification](./multilayerperceptron)
* [Restricted Boltzmann Machines (RBM)](./restrictedboltzmannmachine.html)
* [Eigenvectors, Eigenvalues, PCA, Covariance and Entropy](./eigenvector.html)
* [Markov Chain Monte Carlo and Reinforcement Learning](./markovchainmontecarlo)
* [Graph Data and Deep Learning](./graphdata)
* [Symbolic Reasoning and Deep Learning](./symbolicreasoning)
* [MNIST for Beginners](./mnist-for-beginners.html)
* [Glossary of Deep-Learning and Neural-Net Terms](./glossary)
* [Word2vec and Natural-Language Processing](./word2vec)
* [Deeplearning4j Examples via Quickstart](./quickstart)
* [Neural Networks Demystified](https://www.youtube.com/watch?v=bxe2T-V8XRs) (A seven-video series)
* [Inference: Machine Learning Model Server](./modelserver)

*Thanks to deeplearning.net and [Luke de Oliveira](https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2) for many of these links and dataset descriptions. Any suggestions of open data sets we should include for the Deeplearning4j community are welcome!*
