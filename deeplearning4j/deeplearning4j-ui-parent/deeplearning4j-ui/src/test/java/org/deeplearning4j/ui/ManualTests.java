/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.ui;

import org.apache.commons.io.IOUtils;
import org.datavec.image.loader.LFWLoader;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.deeplearning4j.nn.conf.weightnoise.DropConnect;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.weights.ConvolutionalIterationListener;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.UUID;

import static org.junit.Assert.fail;

/**
 * Test environment for building/debugging UI.
 *
 * Please, do NOT remove @Ignore annotation
 *
 * @author raver119@gmail.com
 */
@Ignore
public class ManualTests {

    private static Logger log = LoggerFactory.getLogger(ManualTests.class);

    @Test
    public void testLaunch() throws Exception {

        //        UiServer server = UiServer.getInstance();
        //
        //        System.out.println("http://localhost:" + server.getPort()+ "/");

        Thread.sleep(10000000000L);

        new ScoreIterationListener(100);
        fail("not implemneted");
    }


    @Test
    public void testTsne() throws Exception {
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        Nd4j.getRandom().setSeed(123);
        BarnesHutTsne b = new BarnesHutTsne.Builder().stopLyingIteration(10).setMaxIter(10).theta(0.5).learningRate(500)
                        .useAdaGrad(true).build();

        org.nd4j.linalg.io.ClassPathResource resource = new org.nd4j.linalg.io.ClassPathResource("/mnist2500_X.txt");
        File f = resource.getTempFileFromArchive();
        INDArray data = Nd4j.readNumpy(f.getAbsolutePath(), "   ").get(NDArrayIndex.interval(0, 100),
                        NDArrayIndex.interval(0, 784));



        org.nd4j.linalg.io.ClassPathResource labels = new org.nd4j.linalg.io.ClassPathResource("mnist2500_labels.txt");
        List<String> labelsList = IOUtils.readLines(labels.getInputStream()).subList(0, 100);
        b.fit(data);
        File save = new File(System.getProperty("java.io.tmpdir"), "labels-" + UUID.randomUUID().toString());
        System.out.println("Saved to " + save.getAbsolutePath());
        save.deleteOnExit();
        b.saveAsFile(labelsList, save.getAbsolutePath());

        INDArray output = b.getData();
        System.out.println("Coordinates");

        UIServer server = UIServer.getInstance();
        Thread.sleep(10000000000L);
    }

    /**
     * This test is for manual execution only, since it's here just to get working CNN and visualize it's layers
     *
     * @throws Exception
     */
    @Test
    public void testCNNActivationsVisualization() throws Exception {
        final int numRows = 40;
        final int numColumns = 40;
        int nChannels = 3;
        int outputNum = LFWLoader.NUM_LABELS;
        int numSamples = LFWLoader.NUM_IMAGES;
        boolean useSubset = false;
        int batchSize = 200;// numSamples/10;
        int iterations = 5;
        int splitTrainNum = (int) (batchSize * .8);
        int seed = 123;
        int listenerFreq = iterations / 5;
        DataSet lfwNext;
        SplitTestAndTrain trainTest;
        DataSet trainInput;
        List<INDArray> testInput = new ArrayList<>();
        List<INDArray> testLabels = new ArrayList<>();

        log.info("Load data....");
        DataSetIterator lfw = new LFWDataSetIterator(batchSize, numSamples, new int[] {numRows, numColumns, nChannels},
                        outputNum, useSubset, true, 1.0, new Random(seed));

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(seed)
                        .activation(Activation.RELU).weightInit(WeightInit.XAVIER)
                        .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                        .updater(new AdaGrad(0.01)).weightNoise(new DropConnect(0.5)).list()
                        .layer(0, new ConvolutionLayer.Builder(4, 4).name("cnn1").nIn(nChannels).stride(1, 1).nOut(20)
                                        .build())
                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2})
                                        .name("pool1").build())
                        .layer(2, new ConvolutionLayer.Builder(3, 3).name("cnn2").stride(1, 1).nOut(40).build())
                        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2})
                                        .name("pool2").build())
                        .layer(4, new ConvolutionLayer.Builder(3, 3).name("cnn3").stride(1, 1).nOut(60).build())
                        .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2, 2})
                                        .name("pool3").build())
                        .layer(6, new ConvolutionLayer.Builder(2, 2).name("cnn3").stride(1, 1).nOut(80).build())
                        .layer(7, new DenseLayer.Builder().name("ffn1").nOut(160).dropOut(0.5).build())
                        .layer(8, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                        .nOut(outputNum).activation(Activation.SOFTMAX).build())

                        .setInputType(InputType.convolutional(numRows, numColumns, nChannels));

        MultiLayerNetwork model = new MultiLayerNetwork(builder.build());
        model.init();

        log.info("Train model....");

        model.setListeners(new ScoreIterationListener(listenerFreq), new ConvolutionalIterationListener(listenerFreq));

        while (lfw.hasNext()) {
            lfwNext = lfw.next();
            lfwNext.scale();
            trainTest = lfwNext.splitTestAndTrain(splitTrainNum, new Random(seed)); // train set that is the result
            trainInput = trainTest.getTrain(); // get feature matrix and labels for training
            testInput.add(trainTest.getTest().getFeatures());
            testLabels.add(trainTest.getTest().getLabels());
            model.fit(trainInput);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(lfw.getLabels());
        for (int i = 0; i < testInput.size(); i++) {
            INDArray output = model.output(testInput.get(i));
            eval.eval(testLabels.get(i), output);
        }
        INDArray output = model.output(testInput.get(0));
        eval.eval(testLabels.get(0), output);
        log.info(eval.stats());
        log.info("****************Example finished********************");

    }

    @Test
    public void testWord2VecPlot() throws Exception {
        File inputFile = new ClassPathResource("/big/raw_sentences.txt").getFile();
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(5).iterations(2).batchSize(1000).learningRate(0.025)
                        .layerSize(100).seed(42).sampling(0).negativeSample(0).windowSize(5)
                        .modelUtils(new BasicModelUtils<VocabWord>()).useAdaGrad(false).iterate(iter).workers(10)
                        .tokenizerFactory(t).build();

        vec.fit();

        //        UiConnectionInfo connectionInfo = UiServer.getInstance().getConnectionInfo();

        //        vec.getLookupTable().plotVocab(100, connectionInfo);

        Thread.sleep(10000000000L);
        fail("Not implemented");
    }

    @Test
    public void testImage() throws Exception {
        INDArray array = Nd4j.create(11, 13);
        for (int i = 0; i < array.rows(); i++) {
            array.putRow(i, Nd4j.create(new double[] {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f,
                            1.2f, 1.3f}));
        }
        writeImage(array, new File("test.png"));
    }

    private void writeImage(INDArray array, File file) {
        //        BufferedImage image = ImageLoader.toImage(array);

        log.info("Array.rank(): " + array.rank());
        log.info("Size(-1): " + array.size(-1));
        log.info("Size(-2): " + array.size(-2));
        BufferedImage imageToRender = new BufferedImage(array.columns(), array.rows(), BufferedImage.TYPE_BYTE_GRAY);
        for (int x = 0; x < array.columns(); x++) {
            for (int y = 0; y < array.rows(); y++) {
                log.info("x: " + (x) + " y: " + y);
                imageToRender.getRaster().setSample(x, y, 0, (int) (255 * array.getRow(y).getDouble(x)));
            }
        }

        try {
            ImageIO.write(imageToRender, "png", file);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    @Test
    public void testCNNActivations2() throws Exception {

        int nChannels = 1;
        int outputNum = 10;
        int batchSize = 64;
        int nEpochs = 10;
        int seed = 123;

        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, 12345);

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(seed)
                        .l2(0.0005)
                        .weightInit(WeightInit.XAVIER)
                        .updater(new Nesterovs(0.01, 0.9)).list()
                        .layer(0, new ConvolutionLayer.Builder(5, 5)
                                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                                        .nIn(nChannels).stride(1, 1).nOut(20).activation(Activation.IDENTITY).build())
                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                        .stride(2, 2).build())
                        .layer(2, new ConvolutionLayer.Builder(5, 5)
                                        //Note that nIn needed be specified in later layers
                                        .stride(1, 1).nOut(50).activation(Activation.IDENTITY).build())
                        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                        .stride(2, 2).build())
                        .layer(4, new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build())
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                        .nOut(outputNum).activation(Activation.SOFTMAX).build())
                        .setInputType(InputType.convolutional(28, 28, nChannels));

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        /*
        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
            .averagingFrequency(1)
            .prefetchBuffer(12)
            .workers(2)
            .reportScoreAfterAveraging(false)
            .useLegacyAveraging(false)
            .build();
        */

        log.info("Train model....");
        model.setListeners(new ConvolutionalIterationListener(1));

        //((NativeOpExecutioner) Nd4j.getExecutioner()).getLoop().setOmpNumThreads(8);

        long timeX = System.currentTimeMillis();
        //        nEpochs = 2;
        for (int i = 0; i < nEpochs; i++) {
            long time1 = System.currentTimeMillis();
            model.fit(mnistTrain);
            //wrapper.fit(mnistTrain);
            long time2 = System.currentTimeMillis();
            log.info("*** Completed epoch {}, Time elapsed: {} ***", i, (time2 - time1));
        }
        long timeY = System.currentTimeMillis();

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        while (mnistTest.hasNext()) {
            DataSet ds = mnistTest.next();
            INDArray output = model.output(ds.getFeatures(), false);
            eval.eval(ds.getLabels(), output);
        }
        log.info(eval.stats());
        mnistTest.reset();

        log.info("****************Example finished********************");
    }

    @Test
    public void testCNNActivationsFrozen() throws Exception {

        int nChannels = 1;
        int outputNum = 10;
        int batchSize = 64;
        int nEpochs = 10;
        int seed = 123;

        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(seed)
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.01, 0.9)).list()
                .layer(0, new FrozenLayer(new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(nChannels).stride(1, 1).nOut(20).activation(Activation.IDENTITY).build()))
                .layer(1, new FrozenLayer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                        .stride(2, 2).build()))
                .layer(2, new FrozenLayer(new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build()))
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum).activation(Activation.SOFTMAX).build())
                .setInputType(InputType.convolutionalFlat(28, 28, nChannels));

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Train model....");
        model.setListeners(new ConvolutionalIterationListener(1));

        for (int i = 0; i < nEpochs; i++) {
            model.fit(mnistTrain);
        }
    }
}
