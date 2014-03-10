---
title: 
layout: default
---


# data set downloads

Below are preserialized datasets that can be downloaded directly for use with deeplearning4j. Preserialized means they're in the correct format for ingestion. Here's how they can be loaded:


             DataSet d = new DataSet();
             BufferedInputStream bis = new BufferedInputStream(new FileInputStream(new File("path/to/your/file")));
             d.load(bis);
             bis.close();

[Mnist Dataset](https://drive.google.com/file/d/0B-O_wola53IsWDhCSEtJWXUwTjg/edit?usp=sharing)