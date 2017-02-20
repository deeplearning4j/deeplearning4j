---
title: 从CSV文件加载数据
layout: cn-default
---

# 从CSV文件加载数据

本页将介绍如何把来自CSV文件的数据加载到神经网络中，这是很有用的信息，尤其是在处理时间序列数据时。Deeplearning4j中有一种比较简单的方法：

    public static void main(String[] args) throws  Exception {
        RecordReader recordReader = new CSVRecordReader(0,",");
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        //读取器、标签索引、可能的标签数量
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,4,3);
        //用记录读取器读取数据集信息。datasetiterator负责向量化
        DataSet next = iterator.next();
        //自定义参数
        Nd4j.MAX_SLICES_TO_PRINT = 10;
        Nd4j.MAX_ELEMENTS_PER_SLICE = 10;

然后就可以配置神经网络，用这个数据集来定型。[具体方法参见示例](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataExamples/CSVExample.java)。 

CSVRecordReader类的详情参见[此处](https://github.com/deeplearning4j/Canova/blob/master/canova-api/src/main/java/org/canova/api/records/reader/impl/CSVRecordReader.java)。
