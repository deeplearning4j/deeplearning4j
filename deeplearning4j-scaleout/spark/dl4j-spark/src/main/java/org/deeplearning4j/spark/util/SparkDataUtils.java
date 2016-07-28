package org.deeplearning4j.spark.util;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.deeplearning4j.spark.datavec.RecordReaderFunction;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by Alex on 15/06/2016.
 */
public class SparkDataUtils {

    private JavaRDD<DataSet> loadFromTextFile(String path, int labelIndex, int numClasses, RecordReader recordReader, JavaSparkContext sc){
        JavaRDD<String> lines = sc.textFile(path);
        // gotta map this to a Matrix/INDArray
        return lines.map(new RecordReaderFunction(recordReader, labelIndex, numClasses));
    }

}
