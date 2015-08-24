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

package org.deeplearning4j.spark.impl.layer;

import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.canova.api.records.reader.RecordReader;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.spark.canova.RecordReaderFunction;
import org.deeplearning4j.spark.impl.common.Add;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import parquet.org.slf4j.Logger;
import parquet.org.slf4j.LoggerFactory;

import java.io.Serializable;

/**
 * Master class for org.deeplearning4j.spark
 * layers
 * @author Adam Gibson
 */
public class SparkDl4jLayer implements Serializable {

    private transient SparkContext sparkContext;
    private transient JavaSparkContext sc;
    private NeuralNetConfiguration conf;
    private Layer layer;
    private Broadcast<INDArray> params;
    private boolean averageEachIteration = false;
    private static Logger log = LoggerFactory.getLogger(SparkDl4jLayer.class);


    public SparkDl4jLayer(SparkContext sparkContext, NeuralNetConfiguration conf) {
        this.sparkContext = sparkContext;
        this.conf = conf.clone();
        sc = new JavaSparkContext(this.sparkContext);
    }

    public SparkDl4jLayer(JavaSparkContext sc, NeuralNetConfiguration conf) {
        this.sc = sc;
        this.conf = conf.clone();
    }

    /**
     * Fit the layer based on the specified org.deeplearning4j.spark context text file
     * @param path the path to the text file
     * @param labelIndex the index of the label
     * @param recordReader the record reader
     * @return the fit layer
     */
    public Layer fit(String path,int labelIndex,RecordReader recordReader) {
        FeedForwardLayer ffLayer = (FeedForwardLayer) conf.getLayer();

        JavaRDD<String> lines = sc.textFile(path);
        // gotta map this to a Matrix/INDArray
        JavaRDD<DataSet> points = lines.map(new RecordReaderFunction(recordReader
                , labelIndex, ffLayer.getNOut()));
        return fitDataSet(points);

    }

    /**
     * Fit the given rdd given the context.
     * This will convert the labeled points
     * to the internal dl4j format and train the model on that
     * @param sc the org.deeplearning4j.spark context
     * @param rdd the rdd to fitDataSet
     * @return the multi layer network that was fitDataSet
     */
    public Layer fit(JavaSparkContext sc,JavaRDD<LabeledPoint> rdd) {
        FeedForwardLayer ffLayer = (FeedForwardLayer) conf.getLayer();
        return fitDataSet(MLLibUtil.fromLabeledPoint(sc, rdd, ffLayer.getNOut()));
    }

    /**
     * Fit a java rdd of dataset
     * @param rdd the rdd to fit
     * @return the fit layer
     */
    public Layer fitDataSet(JavaRDD<DataSet> rdd) {
        int iterations = conf.getNumIterations();
        long count = rdd.count();
        int batchSize = conf.getBatchSize();
        if(batchSize == 0)
            batchSize = 10;

        log.info("Running distributed training averaging each iteration " + averageEachIteration + " and " + rdd.partitions().size() + " partitions");
        if(!averageEachIteration) {
            Layer layer = LayerFactories.getFactory(conf.getLayer()).create(conf);
            final INDArray params = layer.params();
            this.params = sc.broadcast(params);
            log.info("Broadcasting initial parameters of length " + params.length());
            int paramsLength = layer.numParams();
            if(params.length() != paramsLength)
                throw new IllegalStateException("Number of params " + paramsLength + " was not equal to " + params.length());
            JavaRDD<INDArray> results = rdd.sample(true,0.4).mapPartitions(new IterativeReduceFlatMap(conf.toJson(), this.params));
            log.debug("Ran iterative reduce...averaging results now.");
            INDArray newParams = results.fold(Nd4j.zeros(results.first().shape()),new Add());
            newParams.divi(rdd.partitions().size());
            layer.setParams(newParams);
            this.layer = layer;
        }
        else {
            conf.setNumIterations(1);
            Layer layer = LayerFactories.getFactory(conf.getLayer()).create(conf);
            final INDArray params = layer.params();
            this.params = sc.broadcast(params);

            for(int i = 0; i < iterations; i++) {
                JavaRDD<INDArray> results = rdd.sample(true,0.3).mapPartitions(new IterativeReduceFlatMap(conf.toJson(), this.params));

                int paramsLength = layer.numParams();
                if(params.length() != paramsLength)
                    throw new IllegalStateException("Number of params " + paramsLength + " was not equal to " + params.length());

                INDArray newParams = results.fold(Nd4j.zeros(results.first().shape()), new Add());
                newParams.divi(rdd.partitions().size());
            }

            layer.setParams(this.params.value());
            this.layer = layer;


        }


        return layer;
    }


    /**
     * Predict the given feature matrix
     * @param features the given feature matrix
     * @return the predictions
     */
    public Matrix predict(Matrix features) {
        return MLLibUtil.toMatrix(layer.activate(MLLibUtil.toMatrix(features)));
    }


    /**
     * Predict the given vector
     * @param point the vector to predict
     * @return the predicted vector
     */
    public Vector predict(Vector point) {
        return MLLibUtil.toVector(layer.activate(MLLibUtil.toVector(point)));
    }


    /**
     * Train a multi layer network
     * @param data the data to train on
     * @param conf the configuration of the network
     * @return the fit multi layer network
     */
    public static Layer train(JavaRDD<LabeledPoint> data,NeuralNetConfiguration conf) {
        SparkDl4jLayer multiLayer = new SparkDl4jLayer(data.context(),conf);
        return multiLayer.fit(new JavaSparkContext(data.context()),data);

    }



}
