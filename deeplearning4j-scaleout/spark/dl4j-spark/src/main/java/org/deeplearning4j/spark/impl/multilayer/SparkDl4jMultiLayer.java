package org.deeplearning4j.spark.impl.multilayer;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.canova.api.records.reader.RecordReader;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.canova.RecordReaderFunction;
import org.deeplearning4j.spark.impl.common.Add;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;

/**
 * Master class for spark
 *
 * @author Adam Gibson
 */
public class SparkDl4jMultiLayer implements Serializable {

    private transient SparkContext sparkContext;
    private transient JavaSparkContext sc;
    private MultiLayerConfiguration conf;
    private MultiLayerNetwork network;
    private Broadcast<INDArray> params;
    private boolean averageEachIteration = false;
    public final static String AVERAGE_EACH_ITERATION = "org.deeplearning4j.spark.iteration.average";
    private static final Logger log = LoggerFactory.getLogger(SparkDl4jMultiLayer.class);

    /**
     * Instantiate a multi layer spark instance
     * with the given context and network.
     * This is the prediction constructor
     * @param sparkContext  the spark context to use
     * @param network the network to use
     */
    public SparkDl4jMultiLayer(SparkContext sparkContext, MultiLayerNetwork network) {
        this.sparkContext = sparkContext;
        this.averageEachIteration = sparkContext.conf().getBoolean(AVERAGE_EACH_ITERATION,false);
        this.network = network;
        this.conf = this.network.getLayerWiseConfigurations().clone();
        sc = new JavaSparkContext(this.sparkContext);
    }

    /**
     * Training constructor. Instantiate with a configuration
     * @param sparkContext the spark context to use
     * @param conf the configuration of the network
     */
    public SparkDl4jMultiLayer(SparkContext sparkContext, MultiLayerConfiguration conf) {
        this.sparkContext = sparkContext;
        this.conf = conf.clone();
        this.averageEachIteration = sparkContext.conf().getBoolean(AVERAGE_EACH_ITERATION,false);
        sc = new JavaSparkContext(this.sparkContext);
    }

    /**
     * Training constructor. Instantiate with a configuration
     * @param sc the spark context to use
     * @param conf the configuration of the network
     */
    public SparkDl4jMultiLayer(JavaSparkContext sc, MultiLayerConfiguration conf) {
        this(sc.sc(),conf);
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

    public MultiLayerNetwork getNetwork() {
        return network;
    }

    public void setNetwork(MultiLayerNetwork network) {
        this.network = network;
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
        int iterations = conf.getConf(0).getNumIterations();
        long count = rdd.count();
        int batchSize = conf.getConf(0).getBatchSize();
        int newBatchSize = (int) (count / batchSize);
        rdd = rdd.repartition(newBatchSize);
        log.info("Running distributed training averaging each iteration " + averageEachIteration + " and " + rdd.partitions().size() + " partitions");
        if(!averageEachIteration) {
            MultiLayerNetwork network = new MultiLayerNetwork(conf);
            network.init();
            final INDArray params = network.params();
            this.params = sc.broadcast(params);

            int paramsLength = network.numParams();
            if(params.length() != paramsLength)
                throw new IllegalStateException("Number of params " + paramsLength + " was not equal to " + params.length());

            JavaRDD<INDArray> results = rdd.mapPartitions(new IterativeReduceFlatMap(conf.toJson(),this.params));
            INDArray newParams = results.fold(Nd4j.zeros(results.first().shape()),new Add());
            newParams.divi(rdd.partitions().size());
            network.setParameters(newParams);
            this.network = network;
        }
        else {
            for(NeuralNetConfiguration conf : this.conf.getConfs())
                conf.setNumIterations(1);
            MultiLayerNetwork network = new MultiLayerNetwork(conf);
            network.init();
            final INDArray params = network.params();
            this.params = sc.broadcast(params);

            for(int i = 0; i < iterations; i++) {


                int paramsLength = network.numParams();
                if(params.length() != paramsLength)
                    throw new IllegalStateException("Number of params " + paramsLength + " was not equal to " + params.length());
                JavaRDD<INDArray> miniBatchParams = rdd.map(new IterativeReduce(this.params,conf.toJson()));

                INDArray add = miniBatchParams.reduce(new Add());
                this.params =  sc.broadcast(add.divi(rdd.partitions().size()));
            }


            network.setParameters(this.params.value());
            this.network = network;


        }


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
