---
title: 
layout: default
---

# dataset downloads

DL4J works with three kinds of downloadable datasets. 

1. The first are preserialized datasets that can be downloaded directly for use with DL4J neural nets. Preserialized means they're in the correct format for ingestion. You can load them to RAM without using a dataset iterator — you don’t have to create them. The [Mnist Dataset](https://drive.google.com/file/d/0B-O_wola53IsWDhCSEtJWXUwTjg/edit?usp=sharing) is a good example. 

Here's how they can be loaded:


             DataSet d = new DataSet();
             BufferedInputStream bis = new BufferedInputStream(new FileInputStream(new File("path/to/your/file")));
             d.load(bis);
             bis.close();

2. The second dataset type is pre-compiled binaries, or jar files. These consist of the actual classes themselves, and can be found on [Maven](http://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j). With the binaries, you run everything. 

3. The third type of dataset, which we will upload shortly, is a light-weight, pre-trained neural net that you can simply call load on. 