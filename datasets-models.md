---
title: "Datasets & Models"
layout: default
---

# Datasets & Models

We'll talk about two types of downloads here.  

### datasets

The first are preserialized datasets that can be downloaded directly for use with DL4J neural nets. Preserialized means they're in the correct format for ingestion. You can load them to RAM without using a dataset iterator — you don’t have to create them.  Here's how they can be loaded:

             DataSet d = new DataSet();
             BufferedInputStream bis = new BufferedInputStream(new FileInputStream(new File("path/to/your/file")));
             d.load(bis);
             bis.close();

### models

Preserialized models, which we will upload shortly, are light-weight, pre-trained neural nets that you can simply call load on. 
