package org.deeplearning4j.spark.impl.multilayer;



import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.canova.api.records.reader.impl.SVMLightRecordReader;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.BaseSparkTest;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.junit.Test;
import org.nd4j.linalg.api.activation.Activations;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.List;

import static org.junit.Assert.assertEquals;


/**
 * Created by agibsonccc on 1/18/15.
 */
public class TestSparkMultiLayer extends BaseSparkTest {

    private static Logger log = LoggerFactory.getLogger(TestSparkMultiLayer.class);


    @Test
    public void testIris() throws Exception {


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().lossFunction(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                .nIn(4).nOut(3).layerFactory(LayerFactories.getFactory(RBM.class)).batchSize(50).visibleUnit(RBM.VisibleUnit.GAUSSIAN)
                .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
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


    @Test
    public void testStaticInvocation() {
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

        DataSet dataSet = new IrisDataSetIterator(150,150).next();
        List<DataSet> list = dataSet.asList();
        JavaRDD<DataSet> data = sc.parallelize(list);
        JavaRDD<LabeledPoint> mllLibData = MLLibUtil.fromDataSet(sc,data);

        MultiLayerNetwork network = SparkDl4jMultiLayer.train(mllLibData,conf);


    }



}
