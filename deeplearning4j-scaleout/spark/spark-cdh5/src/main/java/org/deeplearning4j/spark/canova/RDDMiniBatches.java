package org.deeplearning4j.spark.canova;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.rdd.RDD;
import org.deeplearning4j.spark.ordering.DataSetOrdering;
import org.nd4j.linalg.dataset.DataSet;

/**
 * Created by agibsonccc on 1/18/15.
 */
public class RDDMiniBatches {
    private int miniBatches = 10;
    private RDD<DataSet> toSplit;
    private JavaRDD<DataSet> toSplitJava;
    public RDDMiniBatches(int miniBatches, RDD<DataSet> toSplit) {
        this.miniBatches = miniBatches;
        this.toSplit = toSplit;
    }

    public RDDMiniBatches(int miniBatches, JavaRDD<DataSet> toSplit) {
        this.miniBatches = miniBatches;
        this.toSplitJava = toSplit;
    }

    public JavaRDD<DataSet> miniBatchesJava() {
        long count = toSplit.count();
        int batchSize = miniBatches;
        int numBatches =(int) count / batchSize;
        JavaRDD<DataSet> miniBatches = toSplitJava.coalesce(numBatches,true);
        return miniBatches;
    }

    public RDD<DataSet> miniBatches() {
        long count = toSplit.count();
        int batchSize = miniBatches;
        int numBatches =(int) count / batchSize;
        RDD<DataSet> miniBatches = toSplit.coalesce(numBatches,true,new DataSetOrdering());
        return miniBatches;
    }


}
