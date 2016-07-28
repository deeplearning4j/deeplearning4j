package org.deeplearning4j.streaming.pipeline.spark.inerference;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by agibsonccc on 6/11/16.
 */
public class PrintDataSet implements Function<JavaRDD<INDArray>, Void> {
    @Override
    public Void call(JavaRDD<INDArray> dataSetJavaRDD) throws Exception {
        dataSetJavaRDD.foreach(new VoidFunction<INDArray>() {
            @Override
            public void call(INDArray dataSet) throws Exception {
                System.out.println(dataSet);
            }
        });

        return null;
    }
}

