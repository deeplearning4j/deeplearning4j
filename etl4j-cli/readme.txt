How to use Canova (generally):

./bin/canova vectorize -conf <conf_file>

If you just type the command, the full option list will print out.


Examples:


CSV: UCI Iris

./bin/canova vectorize -conf examples/csv/UCI_Iris/conf/csv_conf.txt 


Text:

Children's Books Example

./bin/canova vectorize -conf examples/text/ChildrensBooks/conf/childrens_book_conf.txt 

Tweets Example

./bin/canova vectorize -conf examples/text/Tweets/conf/tweet_conf.txt 


Image: LFW Dataset

Steps to setup example:

1. download lfw dataset locally 

http://vis-www.cs.umass.edu/lfw/lfw.tgz

2. untar the file into a directory

3. configure the canova.input.directory property to be set to this base directory



Custom: MNIST Dataset

Steps to setup example:

1. download mnist dataset locally (labels and images)

http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

2. Put the two files in a common subdirectory and unzip them:

ls -la /tmp/MNIST/
total 111432
drwxr-xr-x   6 josh  wheel       204 Jun 26 15:44 .
drwxrwxrwt  17 root  wheel       578 Jun 26 15:44 ..
-rw-r--r--   1 josh  wheel  47040016 Jun 26 15:44 images-idx1-ubyte
-rw-r--r--   1 josh  wheel   9912422 Jun 26 15:44 images-idx1-ubyte.gz
-rw-r--r--   1 josh  wheel     60008 Jun 26 15:44 labels-idx1-ubyte
-rw-r--r--   1 josh  wheel     28881 Jun 26 15:44 labels-idx1-ubyte.gz

3. configure the canova.input.directory property to be set to the images file (the input format will find the labels file)

canova.input.directory=/tmp/MNIST/images-idx1-ubyte

4. Run Canova from bash

./bin/canova vectorize -conf examples/mnist/conf/mnist_conf.txt 

