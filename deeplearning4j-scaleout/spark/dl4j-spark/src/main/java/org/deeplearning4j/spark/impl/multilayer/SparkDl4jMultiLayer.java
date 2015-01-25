package org.deeplearning4j.spark.impl.multilayer;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.canova.api.records.reader.RecordReader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.canova.RDDMiniBatches;
import org.deeplearning4j.spark.canova.RecordReaderFunction;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.Serializable;

/**
 * Master class for org.deeplearning4j.spark
 *
 * @author Adam Gibson
 */
public class SparkDl4jMultiLayer implements Serializable {

    private transient SparkContext sparkContext;
    private transient JavaSparkContext sc;
    private MultiLayerConfiguration conf;
    private MultiLayerNetwork network;

    public SparkDl4jMultiLayer(SparkContext sparkContext, MultiLayerNetwork network) {
        this.sparkContext = sparkContext;
        this.conf = conf.clone();
        sc = new JavaSparkContext(this.sparkContext);
        this.network = network;
    }

    public SparkDl4jMultiLayer(SparkContext sparkContext, MultiLayerConfiguration conf) {
        this.sparkContext = sparkContext;
        this.conf = conf.clone();
        sc = new JavaSparkContext(this.sparkContext);
    }

    public SparkDl4jMultiLayer(JavaSparkContext sc, MultiLayerConfiguration conf) {
        this.sc = sc;
        this.conf = conf.clone();
    }

    /**
     * Train a multi layer network based on the path
     * @param path the path to the text file
     * @param labelIndex the label index
     * @param recordReader the record reader to parse results
     * @return
     */
    public MultiLayerNetwork fit(String path,int labelIndex,RecordReader recordReader) {
        JavaRDD<String> lines = sc.textFile(path);
        // gotta map this to a Matrix/INDArray
        JavaRDD<DataSet> points = lines.map(new RecordReaderFunction(recordReader
                , labelIndex, conf.getConf(conf.getConfs().size() - 1).getnOut()));
        return fitDataSet(points);

    }

    /**
     * Predict the given feature matrix
     * @param features the given feature matrix
     * @return the predictions
     */
    public Matrix predict(Matrix features) {
        return MLLibUtil.toMatrix(network.output(MLLibUtil.toMatrix(features)));
    }


    /**
     * Predict the given vector
     * @param point the vector to predict
     * @return the predicted vector
     */
    public Vector predict(Vector point) {
        return MLLibUtil.toVector(network.output(MLLibUtil.toVector(point)));
    }


    /**
     * Fit the given rdd given the context.
     * This will convert the labeled points
     * to the internal dl4j format and train the model on that
     * @param sc the org.deeplearning4j.spark context
     * @param rdd the rdd to fitDataSet
     * @return the multi layer network that was fitDataSet
     */
    public MultiLayerNetwork fit(JavaSparkContext sc,JavaRDD<LabeledPoint> rdd) {
        return fitDataSet(MLLibUtil.fromLabeledPoint(sc, rdd, conf.getConf(conf.getConfs().size() - 1).getnOut()));
    }


    /**
     * Fit the dataset rdd
     * @param rdd the rdd to fitDataSet
     * @return the multi layer network
     */
    public MultiLayerNetwork fitDataSet(JavaRDD<DataSet> rdd) {

        int batchSize = conf.getConf(0).getBatchSize();
        JavaRDD<DataSet> miniBatches = new RDDMiniBatches(batchSize,rdd).miniBatchesJava();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        INDArray params = network.params();
        int paramsLength = network.numParams();
        if(params.length() != paramsLength)
            throw new IllegalStateException("Number of params " + paramsLength + " was not equal to " + params.length());
        DL4jWorker worker = new DL4jWorker(conf.toJson(),params);
        INDArray newParams = miniBatches.map(worker)
                .reduce(new Function2<INDArray, INDArray, INDArray>() {
                    @Override
                    public INDArray call(INDArray v1, INDArray v2) throws Exception {
                        return v1.addi(v2);
                    }
                }).divi(miniBatches.count());
        network.setParameters(newParams);
        this.network = network;
        return network;
    }

    /**
     * Train a multi layer network
     * @param data the data to train on
     * @param conf the configuration of the network
     * @return the fit multi layer network
     */
    public static MultiLayerNetwork train(JavaRDD<LabeledPoint> data,MultiLayerConfiguration conf) {
        SparkDl4jMultiLayer multiLayer = new SparkDl4jMultiLayer(data.context(),conf);
        return multiLayer.fit(new JavaSparkContext(data.context()),data);

    }
}
