---
title: 
layout: default
---

# datasets & downloads

We'll talk about three types of downloads here.  

### datasets

The first are preserialized datasets that can be downloaded directly for use with DL4J neural nets. Preserialized means they're in the correct format for ingestion. You can load them to RAM without using a dataset iterator — you don’t have to create them. The [Mnist Dataset](https://drive.google.com/file/d/0B-O_wola53IsWDhCSEtJWXUwTjg/edit?usp=sharing) is a good example. Here's how they can be loaded:

             DataSet d = new DataSet();
             BufferedInputStream bis = new BufferedInputStream(new FileInputStream(new File("path/to/your/file")));
             d.load(bis);
             bis.close();

### code

The second download is pre-compiled binaries, or jar files. These consist of the actual classes themselves, and can be found on [Maven](http://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j). With the binaries, you run everything. 

### models

Preserialized models, which we will upload shortly, are light-weight, pre-trained neural nets that you can simply call load on. 