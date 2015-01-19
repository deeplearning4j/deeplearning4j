package org.deeplearning4j.spark.impl.multilayer;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.nn.api.MultiLayer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.canova.RDDMiniBatches;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

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




    public MultiLayerNetwork fit(JavaRDD<DataSet> rdd) {
        int batchSize = conf.getConf(0).getBatchSize();
        JavaRDD<DataSet> miniBatches = new RDDMiniBatches(batchSize,rdd).miniBatchesJava();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        INDArray newParams = miniBatches.map(new DL4jWorker(network)).reduce(new Function2<INDArray, INDArray, INDArray>() {
            @Override
            public INDArray call(INDArray v1, INDArray v2) throws Exception {
                return v1.add(v2);
            }
        }).divi(miniBatches.count());
        network.setParameters(newParams);
        return network;
    }





}
