package org.deeplearning4j.datasets.fetchers;

import org.apache.commons.io.FilenameUtils;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.split.LimitFileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.base.LFWFetcher;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by nyghtowl on 12/16/15.
 */
public class LfwFetcherTest {

    @Test
    public void testLfwFetcher() {
        File dir = new File(FilenameUtils.concat(System.getProperty("user.home"), "lfw-a"));
        new LFWFetcher("lfw-a", true).fetch();
        assertTrue(dir.exists());

    }

    @Test
    public void testLfwDataFetchter() {
        String subDir = "lfw-a";
        String path = FilenameUtils.concat(System.getProperty("user.home"), subDir);
        LFWDataFetcher lf = new LFWDataFetcher(250, 250, 3, path, true);
        lf.fetch(10);
        assertEquals("Aaron_Eckhart", lf.getLabelName(0));
        assertEquals(432, lf.getNumNames());
    }

    @Test
    public void testLfwReader() throws Exception {
        String subDir = "lfw-a/lfw";
        String path = FilenameUtils.concat(System.getProperty("user.home"), subDir);
        RecordReader rr = new ImageRecordReader(250, 250, 3, true, ".[0-9]+");
        rr.initialize(new LimitFileSplit(new File(path), null, 10, 5, ".[0-9]+", new Random(123)));
        RecordReaderDataSetIterator rrd = new RecordReaderDataSetIterator(rr, 10, 250*250*3, 8);
        assertEquals("Aaron_Sorkin", rr.getLabels().get(0));
    }

    @Test
    public void testLfwModel() throws Exception{
        final int numRows = 250;
        final int numColumns = 250;
        int numChannels = 3;
        int outputNum = 432;
        int numSamples = 10;
        int categories = 8;
        int iterations = 1;
        int seed = 123;
        int listenerFreq = iterations/5;

        // TODO LfwDataSetIterator & example
//        DataSetIterator lfw = new LFWDataSetIterator(batchSize, numSamples, numColumns, numRows, numChannels, true);
        String subDir = "lfw-a/lfw";
        String path = FilenameUtils.concat(System.getProperty("user.home"), subDir);
        RecordReader rr = new ImageRecordReader(numColumns, numRows, numChannels, true, ".[0-9]+");
        rr.initialize(new LimitFileSplit(new File(path), null, numSamples, categories, ".[0-9]+", new Random(123)));
        RecordReaderDataSetIterator lfw = new RecordReaderDataSetIterator(rr, 10, 250*250*3, 8);

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new ConvolutionLayer.Builder(10, 10)
                        .nIn(numChannels)
                        .nOut(6)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[] {2,2})
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(categories)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder,numRows,numColumns,numChannels);

        MultiLayerNetwork model = new MultiLayerNetwork(builder.build());
        model.init();

        model.setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

        model.fit(lfw.next());

        DataSet dataTest = lfw.next();
        INDArray output = model.output(dataTest.getFeatureMatrix());
        Evaluation eval = new Evaluation(outputNum); // TODO pass in labels
        eval.eval(dataTest.getLabels(), output);
        System.out.println(eval.stats());


    }

}
