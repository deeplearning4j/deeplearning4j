---
title:Loading Data From CSV's
layout: default
---

# CSVファイルからのデータの読み込み

CSVファイルからのデータを読み込む方法を知っておくと、時系列を扱う場合などに便利です。Deeplearning4jで行う場合、以下のような簡単な方法があります。

    public static void main(String[] args) throws  Exception {
        RecordReader recordReader = new CSVRecordReader(0,",");
        recordReader.initialize(new FileSplit(new ClassPathResource("iris.txt").getFile()));
        //リーダー、ラベル・インデックス、可能なラベル数
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,4,3);
        //レコード・リーダーを使ってデータセットを得ます。データセット・イテレーターがベクトル化を行います。
        DataSet next = iterator.next();
        // パラメータをカスタム化
        Nd4j.MAX_SLICES_TO_PRINT = 10;
        Nd4j.MAX_ELEMENTS_PER_SLICE = 10;

この次にニューラルネットワークを設定し、データセットでトレーニングします。その方法については[こちら](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/dataexamples/CSVExample.java)を参考にしてください。 

また、CSVRecordReaderクラスは[こちら](https://github.com/deeplearning4j/DataVec/blob/master/datavec-api/src/main/java/org/datavec/api/records/reader/impl/csv/CSVRecordReader.java)で入手できます。
