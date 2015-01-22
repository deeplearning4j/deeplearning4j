package org.deeplearning4j.spark.impl.multilayer;


import static org.junit.Assert.*;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.canova.api.records.reader.impl.SVMLightRecordReader;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.core.io.ClassPathResource;


/**
 * Created by agibsonccc on 1/18/15.
 */
public class TestSparkMultiLayer {

    @Test
    public void testIris() throws Exception {
        // set to test mode
        SparkConf sparkConf = new SparkConf()
                .setMaster("local")
                .setAppName("sparktest");

        System.out.println("Setting up Spark Context...");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .nIn(4).nOut(3).layerFactory(LayerFactories.getFactory(RBM.class)).batchSize(10)
                .activationFunction(Activations.tanh()).list(2).hiddenLayerSizes(3)
                .override(new NeuralNetConfiguration.ConfOverride() {
                    @Override
                    public void override(int i, NeuralNetConfiguration.Builder builder) {
                        if (i == 1) {
                            builder.activationFunction(Activations.softMaxRows());
                            builder.layerFactory(LayerFactories.getFactory(OutputLayer.class));
                            builder.lossFunction(LossFunctions.LossFunction.MCXENT);
                        }
                    }
                }).build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        int numParams = network.numParams();
        INDArray params = network.params();
        assertEquals(numParams,params.length());
        SparkDl4jMultiLayer sparkDl4jMultiLayer = new SparkDl4jMultiLayer(sc,conf);
        String path = new ClassPathResource("data/svmLight/iris_svmLight_0.txt").getFile().toURI().toString();
        sparkDl4jMultiLayer.fit(path,4,new SVMLightRecordReader());



    }


}
