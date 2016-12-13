package org.deeplearning4j.spark.models.sequencevectors.export;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * This interface describes
 *
 * @author raver119@gmail.com
 */
public interface SparkModelExporter<T extends SequenceElement> {

    /**
     * This method will be called at final stage of SequenceVectors training, and JavaRDD being passed as argument will
     *
     * @param rdd
     */
    void export(JavaRDD<ExportContainer<T>> rdd);
}
