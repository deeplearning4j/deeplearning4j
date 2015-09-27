/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

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
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.canova.RecordReaderFunction;
import org.deeplearning4j.spark.impl.common.Adder;
import org.deeplearning4j.spark.impl.common.gradient.GradientAdder;
import org.deeplearning4j.spark.impl.multilayer.gradientaccum.GradientAccumFlatMap;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
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
    public final static String ACCUM_GRADIENT = "org.deeplearning4j.spark.iteration.accumgrad";
    public final static String DIVIDE_ACCUM_GRADIENT = "org.deeplearning4j.spark.iteration.dividegrad";

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
        this.params = sc.broadcast(network.params());
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
     * @return {@link MultiLayerNetwork}
     */
    public MultiLayerNetwork fit(String path,int labelIndex,RecordReader recordReader) {
        JavaRDD<String> lines = sc.textFile(path);
        // gotta map this to a Matrix/INDArray
        FeedForwardLayer outputLayer = (FeedForwardLayer) conf.getConf(conf.getConfs().size() - 1).getLayer();
        JavaRDD<DataSet> points = lines.map(new RecordReaderFunction(recordReader
                , labelIndex, outputLayer.getNOut()));
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
     * @param rdd the rdd to fitDataSet
     * @return the multi layer network that was fitDataSet
     */
    public MultiLayerNetwork fit(JavaRDD<LabeledPoint> rdd,int batchSize) {
        FeedForwardLayer outputLayer = (FeedForwardLayer) conf.getConf(conf.getConfs().size() - 1).getLayer();
        return fitDataSet(MLLibUtil.fromLabeledPoint(rdd, outputLayer.getNOut(),batchSize));
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
        FeedForwardLayer outputLayer = (FeedForwardLayer) conf.getConf(conf.getConfs().size() - 1).getLayer();
        return fitDataSet(MLLibUtil.fromLabeledPoint(sc, rdd, outputLayer.getNOut()));
    }


    /**
     * Fit the dataset rdd
     * @param rdd the rdd to fitDataSet
     * @return the multi layer network
     */
    public MultiLayerNetwork fitDataSet(JavaRDD<DataSet> rdd) {
        int iterations = conf.getConf(0).getNumIterations();
        log.info("Running distributed training averaging each iteration " + averageEachIteration + " and " + rdd.partitions().size() + " partitions");
        if(!averageEachIteration)
              runIteration(rdd);

        else {
            for(NeuralNetConfiguration conf : this.conf.getConfs())
                conf.setNumIterations(1);
            MultiLayerNetwork network = new MultiLayerNetwork(conf);
            network.init();
            final INDArray params = network.params();
            this.params = sc.broadcast(params);

            for(int i = 0; i < iterations; i++)
                runIteration(rdd);

        }


        return network;
    }

    private void runIteration(JavaRDD<DataSet> rdd) {
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        final INDArray params = network.params();
        this.params = sc.broadcast(params);
        log.info("Broadcasting initial parameters of length " + params.length());
        int paramsLength = network.numParams();
        if(params.length() != paramsLength)
            throw new IllegalStateException("Number of params " + paramsLength + " was not equal to " + params.length());
        boolean accumGrad = sc.getConf().getBoolean(ACCUM_GRADIENT,false);
        if(accumGrad) {
            JavaRDD<Gradient> results = rdd.mapPartitions(new GradientAccumFlatMap(conf.toJson(), this.params),true).cache();
            log.info("Ran iterative reduce...averaging results now.");
            GradientAdder a = new GradientAdder(params.length());
            results.foreach(a);
            INDArray accumulatedGradient = a.getAccumulator().value();
            boolean divideGrad = sc.getConf().getBoolean(DIVIDE_ACCUM_GRADIENT,false);
            if(divideGrad)
                accumulatedGradient.divi(results.partitions().size());
            log.info("Accumulated parameters");
            log.info("Summed gradients.");
            network.setParameters(network.params().addi(accumulatedGradient));
            log.info("Set parameters");
            this.network = network;
        }
        else {
            JavaRDD<INDArray> results = rdd.mapPartitions(new IterativeReduceFlatMap(conf.toJson(), this.params),true).cache();
            log.info("Ran iterative reduce...averaging results now.");
            Adder a = new Adder(params.length());
            results.foreach(a);
            INDArray newParams = a.getAccumulator().value();
            log.info("Accumulated parameters");
            newParams.divi(rdd.partitions().size());
            log.info("Divided by partitions");
            network.setParameters(newParams);
            log.info("Set parameters");
            this.network = network;
        }

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
