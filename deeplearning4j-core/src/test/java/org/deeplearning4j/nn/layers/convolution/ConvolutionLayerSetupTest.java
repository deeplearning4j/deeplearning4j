package org.deeplearning4j.nn.layers.convolution;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class ConvolutionLayerSetupTest {

    @Test
    public void testConvolutionLayerSetup() {
        MultiLayerConfiguration.Builder builder = inComplete();
        new ConvolutionLayerSetup(builder,28,28,1);
        MultiLayerConfiguration completed = complete().build();
        MultiLayerConfiguration test = builder.build();
        assertEquals(completed,test);

    }


    @Test
    public void testDenseToOutputLayer() {
        final int numRows = 75;
        final int numColumns = 75;
        int nChannels = 3;
        int outputNum = 6;
        int batchSize = 500;
        int iterations = 10;
        int seed = 123;

        //setup the network
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations).regularization(true)
                .l1(1e-1).l2(2e-4).useDropConnect(true)
                .constrainGradientToUnitNorm(true).miniBatch(true)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nOut(5).dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())

                .layer(1, new SubsamplingLayer
                        .Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .nOut(10).dropOut(0.5)
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(3, new SubsamplingLayer
                        .Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .build())
                .layer(4, new DenseLayer.Builder().nOut(100).activation("relu")
                        .build())

                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);

        new ConvolutionLayerSetup(builder,numRows,numColumns,nChannels);
        DataSet d = new DataSet(Nd4j.rand(10,3,75,75).reshape(10,3 * 75 * 75), FeatureUtil.toOutcomeMatrix(new int[]{1,1,1,1,1,1,1,1,1,1},6));
        MultiLayerNetwork network = new MultiLayerNetwork(builder.build());
        network.init();
        network.fit(d);

    }


    @Test
    public void testMnistLenet() throws Exception {
        MultiLayerConfiguration.Builder incomplete = incompleteMnistLenet();
        ConvolutionLayerSetup setup = new ConvolutionLayerSetup(incomplete,28,28,1);
        //first convolution and subsampling
        assertArrayEquals(new int[]{24,24,20},setup.getOutSizesEachLayer().get("0"));
        assertArrayEquals(new int[]{12,12,20},setup.getOutSizesEachLayer().get("1"));

        //second convolution and subsampling
        assertArrayEquals(new int[]{8,8,50},setup.getOutSizesEachLayer().get("2"));
        assertArrayEquals(new int[]{4,4,50},setup.getOutSizesEachLayer().get("3"));
        assertEquals(800, setup.getnInForLayer().get("4").intValue());
        assertEquals(500, setup.getnInForLayer().get("5").intValue());


        MultiLayerConfiguration testConf = incomplete.build();

        //test instantiation
        DataSetIterator iter = new MnistDataSetIterator(10,10);
        MultiLayerNetwork network = new MultiLayerNetwork(testConf);
        network.init();
        network.fit(iter.next());
    }

    @Test
    public void testMultiChannel() throws Exception {
        //ensure LFW data set is present
        List<String> labels = new ArrayList<>(Arrays.asList("Zico", "Ziwang_Xu"));
        String rootDir = new ClassPathResource("lfwtest").getFile().getAbsolutePath();

        RecordReader reader = new ImageRecordReader(28,28,3,true,labels);
        reader.initialize(new FileSplit(new File(rootDir)));
        DataSetIterator recordReader = new RecordReaderDataSetIterator(reader,28 * 28 * 3,labels.size());
        labels.remove("lfwtest");
        NeuralNetConfiguration.ListBuilder builder = (NeuralNetConfiguration.ListBuilder) incompleteLFW();
        new ConvolutionLayerSetup(builder,28,28,3);
        ConvolutionLayer layer2 = (ConvolutionLayer) builder.getLayerwise().get(2).getLayer();
        assertEquals(6,layer2.getNIn());
        DataSet next = recordReader.next();
        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();
        network.fit(next);

    }

    @Test
    public void testLRN() throws Exception{
        List<String> labels = new ArrayList<>(Arrays.asList("Zico", "Ziwang_Xu"));
        String rootDir = new ClassPathResource("lfwtest").getFile().getAbsolutePath();

        RecordReader reader = new ImageRecordReader(28,28,3,true,labels);
        reader.initialize(new FileSplit(new File(rootDir)));
        DataSetIterator recordReader = new RecordReaderDataSetIterator(reader,28 * 28 * 3,labels.size());
        labels.remove("lfwtest");
        NeuralNetConfiguration.ListBuilder builder = (NeuralNetConfiguration.ListBuilder) incompleteLRN();
        new ConvolutionLayerSetup(builder,28,28,3);
        ConvolutionLayer layer2 = (ConvolutionLayer) builder.getLayerwise().get(3).getLayer();
        assertEquals(6,layer2.getNIn());

    }


    public MultiLayerConfiguration.Builder incompleteLRN() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(3).optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(6)
                .layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{5, 5}).nOut(6)
                        .build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder( new int[]{2, 2}).build())
                .layer(2, new LocalResponseNormalization.Builder().build())
                .layer(3, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{5, 5}).nOut(6)
                        .build())
                .layer(4, new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[]{2, 2}).build())
                .layer(5, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(2).build());
        return builder;
    }


    public MultiLayerConfiguration.Builder incompleteLFW() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(3).optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(5)
                .layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{5, 5}).nOut(6)
                        .build())
                .layer(1, new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder( new int[]{2, 2}).build())
                .layer(2, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{5, 5}).nOut(6)
                        .build())
                .layer(3, new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[]{2, 2}).build())
                .layer(4, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(2).build());
        return builder;
    }




    public MultiLayerConfiguration.Builder incompleteMnistLenet() {
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(3).optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(6)
                .layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{5, 5}).nIn(1).nOut(20)
                        .build())
                .layer(1,new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[]{1,1},new int[]{2,2}).build())
                .layer(2,new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{5,5}).nIn(1).nOut(50)
                        .build())
                .layer(3,new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[]{1,1},new int[]{2,2}).build())
                .layer(4,new org.deeplearning4j.nn.conf.layers.DenseLayer.Builder().nOut(500)
                        .build())
                .layer(5, new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
                        .nOut(10).build());
        return builder;
    }

    public MultiLayerConfiguration mnistLenet() {
        MultiLayerConfiguration builder = new NeuralNetConfiguration.Builder()
                .seed(3).optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(5)
                .layer(0,new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{5,5}).nIn(1).nOut(6)
                        .build())
                .layer(1,new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[]{5,5},new int[]{2,2}).build())
                .layer(2,new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{5,5}).nIn(1).nOut(6)
                        .build())
                .layer(3,new org.deeplearning4j.nn.conf.layers.SubsamplingLayer.Builder(new int[]{5,5},new int[]{2,2}).build())
                .layer(4,new org.deeplearning4j.nn.conf.layers.OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nIn(150).nOut(10).build()).build();
        return builder;
    }

    public MultiLayerConfiguration.Builder inComplete() {
        int nChannels = 1;
        int outputNum = 10;
        int iterations = 10;
        int seed = 123;

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{10, 10}, new int[]{2, 2})
                        .nIn(nChannels)
                        .nOut(6)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);

        return builder;
    }


    public MultiLayerConfiguration.Builder complete() {
        final int numRows = 28;
        final int numColumns = 28;
        int nChannels = 1;
        int outputNum = 10;
        int iterations = 10;
        int seed = 123;

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
                .list(3)
                .layer(0, new org.deeplearning4j.nn.conf.layers.ConvolutionLayer.Builder(new int[]{10, 10}, new int[]{2, 2})
                        .nIn(nChannels)
                        .nOut(6)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, new int[]{2, 2})
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(5 * 5 * 1 * 6) //216
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .inputPreProcessor(0, new FeedForwardToCnnPreProcessor(numRows, numColumns, 1))
                .inputPreProcessor(2, new CnnToFeedForwardPreProcessor(5,5,6))
                .backprop(true).pretrain(false);

        return builder;
    }



}
