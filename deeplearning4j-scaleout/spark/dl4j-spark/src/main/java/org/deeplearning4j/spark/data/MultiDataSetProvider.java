package org.deeplearning4j.spark.data;

import org.apache.spark.SparkContext;
import org.apache.spark.rdd.RDD;
import org.datavec.api.transform.TransformProcess;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**
 * A provider for an {@link MultiDataSet}
 * rdd.
 * @author Adam Gibson
 */
public interface MultiDataSetProvider {


    /**
     * Return an rdd of type dataset
     * @return
     */
    RDD<MultiDataSet> data(SparkContext sparkContext);


    /**
     * (Optional) The transform process
     * for the dataset provider.
     * @return
     */
    TransformProcess transformProcess();

}
