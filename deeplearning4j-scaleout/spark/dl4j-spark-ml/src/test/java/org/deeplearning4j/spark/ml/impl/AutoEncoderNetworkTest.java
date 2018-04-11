package org.deeplearning4j.spark.ml.impl;


import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.SQLContext;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.spark.ml.utils.DatasetFacade;
import org.deeplearning4j.spark.ml.utils.ParamSerializer;
import org.junit.After;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.UUID;


public class AutoEncoderNetworkTest {

    private SparkConf sparkConf = new SparkConf().setAppName("testing").setMaster("local[4]");
    private JavaSparkContext sparkContext = new JavaSparkContext(sparkConf);
    private SQLContext sqlContext = new SQLContext(sparkContext);

    @Test
    public void testNetwork() {
        DatasetFacade df = DatasetFacade.dataRows(sqlContext.read().json("src/test/resources/autoencoders"));
        Pipeline p = new Pipeline().setStages(new PipelineStage[] {
                        getAssembler(new String[] {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}, "features")});
        DatasetFacade part2 = DatasetFacade.dataRows(p.fit(df.get()).transform(df.get()).select("features"));

        AutoEncoder sparkDl4jNetwork = new AutoEncoder().setInputCol("features").setOutputCol("auto_encoded")
                        .setCompressedLayer(2).setTrainingMaster(new ParamHelper())
                        .setMultiLayerConfiguration(getNNConfiguration());

        AutoEncoderModel sm = sparkDl4jNetwork.fit(part2.get());
        MultiLayerNetwork mln = sm.getNetwork();
        Assert.assertNotNull(mln);
    }

    @Test
    public void testAutoencoderSave() throws IOException {
        DatasetFacade df = DatasetFacade.dataRows(sqlContext.read().json("src/test/resources/autoencoders"));
        Pipeline p = new Pipeline().setStages(new PipelineStage[] {
                        getAssembler(new String[] {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}, "features")});
        DatasetFacade part2 = DatasetFacade.dataRows(p.fit(df.get()).transform(df.get()).select("features"));

        AutoEncoder sparkDl4jNetwork = new AutoEncoder().setInputCol("features").setOutputCol("auto_encoded")
                        .setCompressedLayer(2).setTrainingMaster(new ParamHelper())
                        .setMultiLayerConfiguration(getNNConfiguration());

        AutoEncoderModel sm = sparkDl4jNetwork.fit(part2.get());

        String fileName = UUID.randomUUID().toString();
        sm.write().save(fileName);
        AutoEncoderModel spdm = AutoEncoderModel.load(fileName);
        Assert.assertNotNull(spdm);
        Assert.assertNotNull(spdm.transform(part2.get()));

        File file = new File(fileName);
        File file2 = new File(fileName + "_metadata");
        FileUtils.deleteDirectory(file);
        FileUtils.deleteDirectory(file2);
    }

    @After
    public void closeIt() {
        sparkContext.close();
    }

    private MultiLayerConfiguration getNNConfiguration() {
        return new NeuralNetConfiguration.Builder().seed(12345).updater(new Sgd(0.1))
                        .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT).list()
                        .layer(0, new DenseLayer.Builder().nIn(10).nOut(8).build())
                        .layer(1, new DenseLayer.Builder().nIn(8).nOut(5).build())
                        .layer(2, new DenseLayer.Builder().nIn(5).nOut(2).build())
                        .layer(3, new DenseLayer.Builder().nIn(2).nOut(5).build())
                        .layer(4, new DenseLayer.Builder().nIn(5).nOut(8).build())
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nIn(8)
                                        .nOut(10).build())
                        .pretrain(true).backprop(true).build();
    }

    private static VectorAssembler getAssembler(String[] input, String output) {
        return new VectorAssembler().setInputCols(input).setOutputCol(output);
    }

    static public class ParamHelper implements ParamSerializer {

        public ParameterAveragingTrainingMaster apply() {
            return new ParameterAveragingTrainingMaster.Builder(3).averagingFrequency(2).workerPrefetchNumBatches(2)
                            .batchSizePerWorker(2).build();
        }
    }
}
