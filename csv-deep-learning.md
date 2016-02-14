---
title: Loading Data From CSV's
layout: default
---

# Loading Data From CSV's

It’s useful to know how to load data from CSV files into neural nets, especially when dealing with time series. There’s an easy way to do that with Deeplearning4j:

    public static void main(String[] args) throws  Exception {
        RecordReader recordReader = new CSVRecordReader(0,",");
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        //reader,label index,number of possible labels
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,4,3);
        //get the dataset using the record reader. The datasetiterator handles vectorization
        DataSet next = iterator.next();
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = 10;
        Nd4j.MAX_ELEMENTS_PER_SLICE = 10;

Then configure the neural network and train it on the dataset. [This example shows how](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/csv/CSVExample.java). 

And here is the [CSVRecordReader class](https://github.com/deeplearning4j/Canova/blob/master/canova-api/src/main/java/org/canova/api/records/reader/impl/CSVRecordReader.java).
