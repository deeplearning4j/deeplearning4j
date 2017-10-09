---
title: 深度学习开放数据集
layout: cn-default
---

# 深度学习开放数据集

本页为您整理汇总了各类可供机器学习研究使用的高质量数据集。欢迎您为本页列表推荐新的数据集！您还可以在[维基百科](https://en.wikipedia.org/wiki/List_of_datasets_for_machine_learning_research)等其他地方找到类似的数据集一览表。

## 最近添加

* [开源生物特征识别数据](http://openbiometrics.org/)
* [谷歌Audioset](https://research.google.com/audioset/)：包含取自YouTube视频的2,084,320条人工标记的10秒声音片段，数据集本体由632种音频事件类别组成，目前仍在不断扩大。
* [优步20亿行程数据](https://movement.uber.com/cities)：逐步开放20亿次行程的数据。
* [Yelp公开数据集](https://www.yelp.com/dataset)：Yelp商户、评论及用户数据的子集，用于自然语言处理（NLP）。

## 自然图像数据集

* [MNIST：手写数字](http://yann.lecun.com/exdb/mnist/)：最常用的合理性检验数据集，由黑白手写数字图像组成，图像大小为25x25，数字居中显示。MNIST是一项比较简单的任务，通过MNIST测试不一定表明模型本身能有效运作。
* [CIFAR10 / CIFAR100]( http://www.cs.utoronto.ca/~kriz/cifar.html)：32×32自然图像数据集，10或100种类别。目前已不再普遍使用，但还是可以用来进行合理性检验。
* [Caltech 101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)：101类物体的图片。
* [Caltech 256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)：256类物体的图片。
* [STL-10数据集](http://www.stanford.edu/~acoates//stl10/)：一个用于开发无监督特征学习、深度学习、自学习算法的图像识别数据集。与CIFAR-10相似但有些改动。
* [街景门牌号码（SVHN）数据集](http://ufldl.stanford.edu/housenumbers/): 来自谷歌街景的门牌号码图像，可将其视作自然的循环式MNIST数据集。
* [NORB](http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/)：以不同照明及摆放方式摄制的玩具模型的双目图像。
* [Pascal VOC](http://pascallin.ecs.soton.ac.uk/challenges/VOC/)：通用图像分割/分类数据集，对建立实际图像标注网络的作用有限，但很适合作为基线。
* [Labelme](http://labelme.csail.mit.edu/Release3.0/browserTools/php/dataset.php)：大型已标注图像数据集。
* [ImageNet](http://image-net.org/)：各类新算法实际使用的图像数据集。ImageNet采用包含1000种类别的WordNet分类层级，而许多图像API公司的REST接口提供的标签似乎都与ImageNet的体系颇为相似。
* [LSUN](http://lsun.cs.princeton.edu/2016/)：用于场景理解和多项辅助任务（房间布局估测、显著性预测等）的竞赛数据集。
* [MS COCO](http://mscoco.org/)：通用图像理解/描述生成的竞赛数据集。
* [COIL 20](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php)：360度旋转拍摄的各类物体图像。
* [COIL100 ](http://www1.cs.columbia.edu/CAVE/software/softlib/coil-100.php)：360度旋转拍摄的各类物体图像。
* [谷歌开放图像数据集](https://research.googleblog.com/2016/09/introducing-open-images-dataset.html)：汇集了900万条图像URL链接，经创作共用协议授权，所有图像“均已用6000多种类别的标签进行标注”。

### 地理空间数据

* [OpenStreetMap](http://wiki.openstreetmap.org/wiki/Planet.osm)：开放授权的数据集，包含整个地球的向量数据。包含美国统计局数据（的较老版本）。
* [Landsat8](https://landsat.usgs.gov/landsat-8)：整个地球表面的卫星照片，每隔数周更新一次。
* [NEXRAD](https://www.ncdc.noaa.gov/data-access/radar-data/nexrad)：  多普勒雷达扫描的美国大气环境数据。

## 人工数据集

* [Arcade Universe](https://github.com/caglar/Arcade-Universe)：－一个人工数据集生成器，图像包括各种电子游戏形象，比如俄罗斯方块中的五连/四连方块。这一生成器基于O. Breleux的bugland数据集生成器。
* 受[BabyAISchool](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/BabyAISchool)的构想启发的一系列数据集
* [BabyAIShapesDatasets](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/BabyAIShapesDatasets)：分辨三种简单的形状
* [BabyAIImageAndQuestionDatasets](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/BabyAIImageAndQuestionDatasets)：一个“问题－图像－回答”数据集
* 为对深度学习架构开展实证评价研究而生成的数据集（[DeepVsShallowComparisonICML2007](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/DeepVsShallowComparisonICML2007)）：
* [MnistVariations](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations)：在MNIST数据集中引入可控变化
* [RectanglesData](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/RectanglesData)：区分宽的和高的长方形
* [ConvexNonConvex](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/ConvexNonConvex)：区分凸多边形和凹多边形
* [BackgroundCorrelation](http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/BackgroundCorrelation)：控制有噪声的MNIST背景中的像素关联程度。

## 脸部图像数据集

* [自然脸部检测（LFW）数据集](http://vis-www.cs.umass.edu/lfw/)：包含13000幅经裁剪的脸部区域图像（采用[Viola-Jones检测框架](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework)），标记了图中人的姓名。数据集中的一部分人有两幅图像，人们常用它训练脸部匹配系统。
* [UMD Faces](http://www.umdfaces.io)：已标注的人脸图像数据集，包括来自8501个人的367920幅脸部图像。
* [CASIA WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html)：包含453453幅人脸图像的数据集，经人脸检测后共识别出超过10575个身份。需要进行一些筛选来提高质量。
* [MS-Celeb-1M](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/)：100万幅世界名人图像。需要进行一些筛选才能在深度神经网络上取得最佳结果。
* [Olivetti](http://www.cs.nyu.edu/~roweis/data.html)：一些人的不同脸部图像。
* [Multi-Pie](http://www.multipie.org/)：CMU的Multi-PIE人脸数据库
* [Face-in-Action](http://www.flintbox.com/public/project/5486/)
* [JACFEE](http://www.humintell.com/jacfee/)：日本人和高加索人不同情绪的脸部表情
* [FERET](http://www.itl.nist.gov/iad/humanid/feret/feret_master.html)：脸部识别技术数据库
* [mmifacedb](http://www.mmifacedb.com/)：MMI脸部表情数据库
* [IndianFaceDatabase](http://vis-www.cs.umass.edu/~vidit/IndianFaceDatabase/)
* [耶鲁人脸数据库](http://vision.ucsd.edu/content/yale-face-database)和[耶鲁人脸数据库B](http://vision.ucsd.edu/~leekc/ExtYaleDatabase/ExtYaleB.html))。

## 视频数据集

* [Youtube-8M](https://research.googleblog.com/2016/09/announcing-youtube-8m-large-and-diverse.html)：用于视频理解研究的大型多样化已标记视频数据集。


## 文本数据集

* [20个新闻组数据集](http://qwone.com/~jason/20Newsgroups/)：分类任务，将出现的词映射至新闻组ID。文本分类的经典数据集之一，通常可以用于纯分类算法的基准测试，或者用于验证任意一种IR/索引算法。
* [路透社新闻数据集](https://archive.ics.uci.edu/ml/datasets/Reuters-21578+Text+Categorization+Collection)：（较老）纯分类用途的新闻电讯文本数据集。常用于教程。
* [Penn Treebank](http://www.cis.upenn.edu/~treebank/)：用于下一词预测或下一字预测。
* [UCI垃圾邮件数据库Spambase](https://archive.ics.uci.edu/ml/datasets/Spambase)：（较老）来自著名的UCI机器学习库的经典垃圾邮件数据集。该数据集经过细致的审编，因此可以作为个性化垃圾邮件筛选学习的基线。
* [广播新闻数据集](http://www.ldc.upenn.edu/Catalog/CatalogEntry.jsp?catalogId=LDC97S44)：用于下一词预测的经典大型文本数据集。
* [文本分类数据集](https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M)：  来自Zhang等人，2015；八个内容丰富的文本分类数据集，可用于新文本分类基线的基准测试。样例大小为120K至3.6M，问题所涉及的类别从两个到14个不等。数据集内容来自DBPedia、亚马逊、Yelp、雅虎和AG。
* [WikiText](http://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/)：取自高质量维基百科文章的大型语言模型语料库，由Salesforce MetaMind进行审编。
* [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)：斯坦福问答数据集——用途广泛的问题回答及阅读理解数据集，每项问题的答案都是一段文本。
* [十亿词数据集](http://www.statmt.org/lm-benchmark/)：大型通用语言模型数据集，常用于训练Word2Vec等词的分布式表示。
* [Common Crawl](http://commoncrawl.org/the-data/)：  万兆字节级的网页爬取数据集——最常用于学习词向量。可通过亚马逊S3免费获取。数据集的内容从万维网爬取获得，因此也可以用作互联网的数据集。
* [谷歌图书Ngram数据集](https://aws.amazon.com/datasets/google-books-ngrams/)：取自谷歌图书的连续词数据，是探索一个词何时开始被广泛使用的简易方法。
* [Yelp公开数据集](https://www.yelp.com/dataset)：Yelp商户、评论及用户数据的子集，用于自然语言处理（NLP）。

### 问答

* [Maluuba新闻问答数据集](https://datasets.maluuba.com/NewsQA)：基于CNN新闻报道的1.2万对问答。
* [Quora问答对](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)：Quora发布的首个数据集，包含副本/语义相似度标签。
* [CMU问答数据集](http://www.cs.cmu.edu/~ark/QA-data/)：用维基百科文章人工生成的事实型问答对，配有难度评级。
* [Maluuba目标导向对话](https://datasets.maluuba.com/Frames)：程序型会话数据集，对话旨在完成一项任务或决策，常用于聊天机器人。
* [bAbi](https://research.fb.com/projects/babi/)：来自Facebook人工智能研究所（FAIR）的综合型阅读理解及问答数据集。
* [儿童图书测试](http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz)：从古腾堡计划提供的儿童图书中提取问答对（问题 + 上下文、答案）作为基线，可以用于问题回答（阅读理解）和事实型查询。

### 情感

* [多领域情感分析数据集](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/)：较老的学术型数据集。
* [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/)：较老且相对较小的二元情感分类数据集。目前的研究论文中多改用更大的数据集来进行基准测试。
* [斯坦福情感Treebank](http://nlp.stanford.edu/sentiment/code.html)：斯坦福的情感数据集，每个句子的解析树的各个节点都有高精度的情感标注。

## 推荐和评价系统

* [Movielens](https://grouplens.org/datasets/movielens/)：来自Movielens网站的电影评价数据，数据集有多个不同大小的版本，最小为演示版，最大为中型规模。
* [Million Song数据集：百万首流行歌曲](https://www.kaggle.com/c/msdchallenge)：Kaggle提供的大型开源数据集，元数据丰富，适合进行混合型推荐系统的实验。
* [Last.fm](http://grouplens.org/datasets/hetrec-2011/)：音乐推荐数据集，可访问基础社交网络及其他类型的元数据，可用于混合型系统。
* [Book-Crossing数据集](http://www.informatik.uni-freiburg.de/~cziegler/BX/)：来自Book-Crossing社区。包括278858位用户对271379本书的1149780项评价。
* [Jester](http://www.ieor.berkeley.edu/~goldberg/jester-data/)：73421位用户对100个笑话的410万项连续评价（-10.00到+10.00）。
* [Netflix Prize](http://www.netflixprize.com/)：Netflix发布了其电影评价数据集的匿名版本；其中包括1亿项评价，共有48万名用户参与评价，每人评价的电影数量为1部到所有17770部不等。首个大型Kaggle式数据挑战赛。由于隐私方面的问题，只能通过非官方渠道获取。

## 网络与图像
* [亚马逊关联购买及评价数据](http://snap.stanford.edu/data/#amazon)：从亚马逊的“购买了该商品的用户还购买了……”部分爬取的数据，以及相关产品的评价数据。适合在互联网中进行推荐系统测试。
* [Friendster社交网络数据集](https://archive.org/details/friendster-dataset-201107)：在转型为游戏网站前，Friendster曾以好友列表的形式公开了103750348名用户的匿名数据。

## 语音数据集

* [2000 HUB5英语数据](https://catalog.ldc.upenn.edu/LDC2002T43)：  英语语音数据集，百度最近的深度语音识别论文中采用了该数据集。
* [LibriSpeech](http://www.openslr.org/12/)：包括文本和语音的有声书数据集。由多位朗读者朗读的有声书录音，总计近500小时，语音清晰，按章节划分，同时包含文本和语音。
* [VoxForge](http://www.voxforge.org/)：清晰的带口音英语语音数据集，可用于提高算法遇到不同口音或语调时的稳健性。
* [TIMIT](https://catalog.ldc.upenn.edu/LDC93S1)：  英语语音识别数据集。
* [CHIME](http://spandh.dcs.shef.ac.uk/chime_challenge/data.html)：嘈杂环境语音识别挑战赛数据集，包含了真实、模拟和清晰的语音数据。真实语音是4位讲话人在超过4个嘈杂地点录制的近9000条录音，模拟语音是用多种环境噪音与语音叠加生成的录音，而清晰语音则是无噪音的录音。
* [TED-LIUM](http://www-lium.univ-lemans.fr/en/content/ted-lium-corpus)：  TED演讲的音频文字稿。1495段TED演讲的录音及其完整转录文字稿。

## 符号化音乐数据集

* [Piano-midi.de：古典钢琴乐曲](http://www.piano-midi.de/)
* [Nottingham：1000多首民歌](http://abc.sourceforge.net/NMD/)
* [MuseData：古典音乐电子乐谱库](http://musedata.stanford.edu/)
* [JSB Chorales：众赞歌四声部合唱乐谱](http://www.jsbchorales.net/index.shtml)

## 其他数据集

* [CMU动作捕捉数据集](http://mocap.cs.cmu.edu/)
* [Brodatz数据集：纹理建模](http://www.ux.uis.no/~tranden/brodatz.html)
* [来自CERN的大型强子对撞机（LHC）的300TB高质量数据](http://opendata.cern.ch/search?ln=en&p=Run2011A+AND+collection%3ACMS-Primary-Datasets+OR+collection%3ACMS-Simulated-Datasets+OR+collection%3ACMS-Derived-Datasets)
* [纽约市出租车数据集](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml)：依据美国信息自由法提出要求后获得的纽约市出租车数据，造成了[隐私问题](https://research.neustar.biz/2014/09/15/riding-with-the-stars-passenger-privacy-in-the-nyc-taxicab-dataset/)。
* [优步FOIL数据集](https://github.com/fivethirtyeight/uber-tlc-foil-response)：依据美国信息自由法（FOIL）提出要求后获得的纽约市的450万次优步行程数据。
* [Criteo点击流数据](http://research.criteo.com/outreach/)：来自欧洲一家大型重定向广告公司的大型互联网广告数据集。

### 卫生与生物数据

* [欧盟传染病监测地图](http://ecdc.europa.eu/en/data-tools/atlas/Pages/atlas.aspx)
* [默克分子活性预测挑战赛](http://www.kaggle.com/c/MerckActivity/data)
* [麝香数据集](https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2))：麝香数据集描述了不同构象的分子。每种分子或者是麝香，或者是非麝香，而这一属性由某一种构象决定。

### 政府与统计数据

* [Data USA：最全面的美国公共数据可视化网站](http://datausa.io)
* [欧盟性别统计数据库](http://eige.europa.eu/gender-statistics)
* [荷兰国家地理信息数据库](http://www.nationaalgeoregister.nl/geonetwork/srv/dut/search#fast=index&from=1&to=50&any_OR_geokeyword_OR_title_OR_keyword=landinrichting*&relation=within)（荷兰语）
* [联合国开发计划署项目](http://open.undp.org/#2016)

* 感谢deeplearning.net和[Luke de Oliveira](https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2)提供多项链接和数据集说明。欢迎提出其他任何值得向Deeplearning4j社区介绍的开数据集！
