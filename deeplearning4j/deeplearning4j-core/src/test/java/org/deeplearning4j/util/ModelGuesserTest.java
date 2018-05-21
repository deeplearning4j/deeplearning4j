package org.deeplearning4j.util;

import org.apache.commons.compress.utils.IOUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.Normalizer;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.UUID;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.assumeNotNull;

/**
 * Created by agibsonccc on 12/29/16.
 */
public class ModelGuesserTest extends BaseDL4JTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();


    @Test
    public void testModelGuessFile() throws Exception {
        ClassPathResource sequenceResource =
                        new ClassPathResource("modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_1_model.h5");
        assertTrue(sequenceResource.exists());
        File f = getTempFile(sequenceResource);
        Model guess1 = ModelGuesser.loadModelGuess(f.getAbsolutePath());
        assumeNotNull(guess1);
        ClassPathResource sequenceResource2 =
                        new ClassPathResource("modelimport/keras/examples/mnist_cnn/mnist_cnn_tf_keras_1_model.h5");
        assertTrue(sequenceResource2.exists());
        File f2 = getTempFile(sequenceResource);
        Model guess2 = ModelGuesser.loadModelGuess(f2.getAbsolutePath());
        assumeNotNull(guess2);

    }

    @Test
    public void testModelGuessInputStream() throws Exception {
        ClassPathResource sequenceResource =
                new ClassPathResource("modelimport/keras/examples/mnist_mlp/mnist_mlp_tf_keras_1_model.h5");
        assertTrue(sequenceResource.exists());
        File f = getTempFile(sequenceResource);

        try (InputStream inputStream = new FileInputStream(f)) {
            Model guess1 = ModelGuesser.loadModelGuess(inputStream);
            assumeNotNull(guess1);
        }

        ClassPathResource sequenceResource2 =
                new ClassPathResource("modelimport/keras/examples/mnist_cnn/mnist_cnn_tf_keras_1_model.h5");
        assertTrue(sequenceResource2.exists());
        File f2 = getTempFile(sequenceResource);

        try (InputStream inputStream = new FileInputStream(f2)) {
            Model guess1 = ModelGuesser.loadModelGuess(inputStream);
            assumeNotNull(guess1);
        }
    }



    @Test
    public void testLoadNormalizersFile() throws Exception {
        MultiLayerNetwork net = getNetwork();

        File tempFile = testDir.newFile("testLoadNormalizersFile.bin");

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
    public void testNormalizerInPlace() throws Exception {
        MultiLayerNetwork net = getNetwork();

        File tempFile = testDir.newFile("testNormalizerInPlace.bin");

        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fit(new DataSet(Nd4j.rand(new int[] {2, 2}), Nd4j.rand(new int[] {2, 2})));
        ModelSerializer.writeModel(net, tempFile, true,normalizer);

        Model model = ModelGuesser.loadModelGuess(tempFile.getAbsolutePath());
        Normalizer<?> normalizer1 = ModelGuesser.loadNormalizer(tempFile.getAbsolutePath());
        assertEquals(model, net);
        assertEquals(normalizer, normalizer1);

    }

    @Test
    public void testLoadNormalizersInputStream() throws Exception {
        MultiLayerNetwork net = getNetwork();

        File tempFile = testDir.newFile("testLoadNormalizersInputStream.bin");

        ModelSerializer.writeModel(net, tempFile, true);

        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fit(new DataSet(Nd4j.rand(new int[] {2, 2}), Nd4j.rand(new int[] {2, 2})));
        ModelSerializer.addNormalizerToModel(tempFile, normalizer);
        Model model = ModelGuesser.loadModelGuess(tempFile.getAbsolutePath());
        try (InputStream inputStream = new FileInputStream(tempFile)) {
            Normalizer<?> normalizer1 = ModelGuesser.loadNormalizer(inputStream);
            assertEquals(model, net);
            assertEquals(normalizer, normalizer1);
        }

    }


    @Test
    public void testModelGuesserDl4jModelFile() throws Exception {
        MultiLayerNetwork net = getNetwork();

        File tempFile = testDir.newFile("testModelGuesserDl4jModelFile.bin");

        ModelSerializer.writeModel(net, tempFile, true);

        MultiLayerNetwork network = (MultiLayerNetwork) ModelGuesser.loadModelGuess(tempFile.getAbsolutePath());
        assertEquals(network.getLayerWiseConfigurations().toJson(), net.getLayerWiseConfigurations().toJson());
        assertEquals(net.params(), network.params());
        assertEquals(net.getUpdater().getStateViewArray(), network.getUpdater().getStateViewArray());

    }

    @Test
    public void testModelGuesserDl4jModelInputStream() throws Exception {
        MultiLayerNetwork net = getNetwork();

        File tempFile = testDir.newFile("testModelGuesserDl4jModelInputStream.bin");

        ModelSerializer.writeModel(net, tempFile, true);

        try (InputStream inputStream = new FileInputStream(tempFile)) {
            MultiLayerNetwork network = (MultiLayerNetwork) ModelGuesser.loadModelGuess(inputStream);
            assumeNotNull(network);
            assertEquals(network.getLayerWiseConfigurations().toJson(), net.getLayerWiseConfigurations().toJson());
            assertEquals(net.params(), network.params());
            assertEquals(net.getUpdater().getStateViewArray(), network.getUpdater().getStateViewArray());
        }
    }


    @Test
    public void testModelGuessConfigFile() throws Exception {
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

    @Test
    public void testModelGuessConfigInputStream() throws Exception {
        ClassPathResource resource = new ClassPathResource("modelimport/keras/configs/cnn_tf_config.json",
                ModelGuesserTest.class.getClassLoader());
        File f = getTempFile(resource);

        try (InputStream inputStream = new FileInputStream(f)) {
            Object conf = ModelGuesser.loadConfigGuess(inputStream);
            assertTrue(conf instanceof MultiLayerConfiguration);
        }

        ClassPathResource sequenceResource = new ClassPathResource("/keras/simple/mlp_fapi_multiloss_config.json");
        File f2 = getTempFile(sequenceResource);

        try (InputStream inputStream = new FileInputStream(f2)) {
            Object sequenceConf = ModelGuesser.loadConfigGuess(inputStream);
            assertTrue(sequenceConf instanceof ComputationGraphConfiguration);
        }


        ClassPathResource resourceDl4j = new ClassPathResource("model.json");
        File fDl4j = getTempFile(resourceDl4j);

        try (InputStream inputStream = new FileInputStream(fDl4j)) {
            Object confDl4j = ModelGuesser.loadConfigGuess(inputStream);
            assertTrue(confDl4j instanceof ComputationGraphConfiguration);
        }

    }


    private File getTempFile(ClassPathResource classPathResource) throws Exception {
        InputStream is = classPathResource.getInputStream();
        File f = testDir.newFile();
        BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(f));
        IOUtils.copy(is, bos);
        bos.flush();
        bos.close();
        return f;
    }

    private MultiLayerNetwork getNetwork() {
        int nIn = 5;
        int nOut = 6;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345).l1(0.01).l2(0.01)
                .updater(new Sgd(0.1)).activation(Activation.TANH).weightInit(WeightInit.XAVIER).list()
                .layer(0, new DenseLayer.Builder().nIn(nIn).nOut(20).build())
                .layer(1, new DenseLayer.Builder().nIn(20).nOut(30).build()).layer(2, new OutputLayer.Builder()
                        .lossFunction(LossFunctions.LossFunction.MSE).nIn(30).nOut(nOut).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        return net;
    }

}
