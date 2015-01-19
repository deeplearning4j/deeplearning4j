package org.deeplearning4j.spark.impl.multilayer;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.rdd.RDD;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.spark.canova.RDDMiniBatches;
import org.deeplearning4j.spark.ordering.DataSetOrdering;
import org.nd4j.linalg.dataset.DataSet;
import scala.math.Ordering;

/**
 * Created by agibsonccc on 1/18/15.
 */
public class Master {

    private SparkContext sparkContext;
    private JavaSparkContext sc;
    private MultiLayerConfiguration conf;

    public Master(SparkContext sparkContext,MultiLayerConfiguration conf) {
        this.sparkContext = sparkContext;
        this.conf = conf;
        sc = new JavaSparkContext(this.sparkContext);
    }




    public void fit(RDD<DataSet> rdd) {
        long count = rdd.count();
        int batchSize = conf.getConf(0).getBatchSize();
        RDD<DataSet> miniBatches = new RDDMiniBatches(batchSize,rdd).miniBatches();


    }





}
