package org.deeplearning4j.spark.data;

import org.apache.spark.SparkContext;
import org.apache.spark.rdd.RDD;
import org.datavec.api.transform.TransformProcess;
import org.nd4j.linalg.dataset.DataSet;

/**
 * A provider for an {@link DataSet}
 * rdd.
 * @author Adam Gibson
 */
public interface DataSetProvider {

    /**
     * Return an rdd of type dataset
     * @return
     */
    RDD<DataSet> data(SparkContext sparkContext);

    /**
     * (Optional) The transform process
     * for the dataset provider.
     * @return
     */
    TransformProcess transformProcess();

}
