package org.deeplearning4j.streaming.pipeline.spark;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by agibsonccc on 6/11/16.
 */
public class PrintDataSet implements Function<JavaRDD<DataSet>, Void> {
    @Override
    public Void call(JavaRDD<DataSet> dataSetJavaRDD) throws Exception {
        dataSetJavaRDD.foreach(new VoidFunction<DataSet>() {
            @Override
            public void call(DataSet dataSet) throws Exception {
                System.out.println(dataSet);
            }
        });

        return null;
    }
}

