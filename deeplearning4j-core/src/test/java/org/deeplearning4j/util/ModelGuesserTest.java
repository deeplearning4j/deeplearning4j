package org.deeplearning4j.util;

import org.apache.commons.compress.utils.IOUtils;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.ModelConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.UUID;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeNotNull;

/**
 * Created by agibsonccc on 12/29/16.
 */
public class ModelGuesserTest {


    @Test
    public void testModelGuess() throws Exception {
        ClassPathResource sequenceResource =
                        new ClassPathResource("modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_model.h5");
        assertTrue(sequenceResource.exists());
        File f = getTempFile(sequenceResource);
        Model guess1 = ModelGuesser.loadModelGuess(f.getAbsolutePath());
        assumeNotNull(guess1);
        ClassPathResource sequenceResource2 =
                        new ClassPathResource("modelimport/keras/examples/mnist_cnn/mnist_cnn_tf_model.h5");
        assertTrue(sequenceResource2.exists());
        File f2 = getTempFile(sequenceResource);
        Model guess2 = ModelGuesser.loadModelGuess(f2.getAbsolutePath());
        assumeNotNull(guess2);

    }



    @Test
    public void testLoadNormalizers() throws Exception {
        int nIn = 5;
        int nOut = 6;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).regularization(true).l1(0.01)
                        .l2(0.01).learningRate(0.1).activation(Activation.TANH).weightInit(WeightInit.XAVIER).list()
                        .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(20).build())
                        .layer(1, new DenseLayer.Builder().nIn(20).nOut(30).build()).layer(2, new OutputLayer.Builder()
                                        .lossFunction(LossFunctions.LossFunction.MSE).nIn(30).nOut(nOut).build())
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        File tempFile = File.createTempFile("tsfs", "fdfsdf");
        tempFile.deleteOnExit();

        ModelSerializer.writeModel(net, tempFile, true);

        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fit(new DataSet(Nd4j.rand(new int[] {2, 2}), Nd4j.rand(new int[] {2, 2})));
        ModelSerializer.addNormalizerToModel(tempFile, normalizer);
        Model model = ModelGuesser.loadModelGuess(tempFile.getAbsolutePath());
        Normalizer<?> normalizer1 = ModelGuesser.loadNormalizer(tempFile.getAbsolutePath());
        assertEquals(model, net);
        assertEquals(normalizer, normalizer1);

    }


    @Test
    public void testModelGuesserDl4jModel() throws Exception {
        int nIn = 5;
        int nOut = 6;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).regularization(true).l1(0.01)
                        .l2(0.01).learningRate(0.1).activation(Activation.TANH).weightInit(WeightInit.XAVIER).list()
                        .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(20).build())
                        .layer(1, new DenseLayer.Builder().nIn(20).nOut(30).build()).layer(2, new OutputLayer.Builder()
                                        .lossFunction(LossFunctions.LossFunction.MSE).nIn(30).nOut(nOut).build())
                        .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        File tempFile = File.createTempFile("tsfs", "fdfsdf");
        tempFile.deleteOnExit();

        ModelSerializer.writeModel(net, tempFile, true);

        MultiLayerNetwork network = (MultiLayerNetwork) ModelGuesser.loadModelGuess(tempFile.getAbsolutePath());
        assertEquals(network.getLayerWiseConfigurations().toJson(), net.getLayerWiseConfigurations().toJson());
        assertEquals(net.params(), network.params());
        assertEquals(net.getUpdater().getStateViewArray(), network.getUpdater().getStateViewArray());

    }


    @Test
    public void testModelGuessConfig() throws Exception {
        ClassPathResource resource = new ClassPathResource("modelimport/keras/configs/cnn_tf_config.json",
                        ModelGuesserTest.class.getClassLoader());
        File f = getTempFile(resource);
        String configFilename = f.getAbsolutePath();
        Object conf = ModelGuesser.loadConfigGuess(configFilename);
        assertTrue(conf instanceof MultiLayerConfiguration);

        ClassPathResource sequenceResource = new ClassPathResource("/keras/simple/mlp_fapi_multiloss_config.json");
        File f2 = getTempFile(sequenceResource);
        Object sequenceConf = ModelGuesser.loadConfigGuess(f2.getAbsolutePath());
        assertTrue(sequenceConf instanceof ComputationGraphConfiguration);



        ClassPathResource resourceDl4j = new ClassPathResource("model.json");
        File fDl4j = getTempFile(resourceDl4j);
        String configFilenameDl4j = fDl4j.getAbsolutePath();
        Object confDl4j = ModelGuesser.loadConfigGuess(configFilenameDl4j);
        assertTrue(confDl4j instanceof ComputationGraphConfiguration);

    }


    private File getTempFile(ClassPathResource classPathResource) throws Exception {
        InputStream is = classPathResource.getInputStream();
        File f = new File(UUID.randomUUID().toString());
        f.deleteOnExit();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(f));
        IOUtils.copy(is, bos);
        bos.flush();
        bos.close();
        return f;
    }

}
