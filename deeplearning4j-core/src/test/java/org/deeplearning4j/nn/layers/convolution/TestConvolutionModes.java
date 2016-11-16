package org.deeplearning4j.nn.layers.convolution;

import org.deeplearning4j.exception.DL4JException;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

/**
 * Created by Alex on 15/11/2016.
 */
public class TestConvolutionModes {

    @Test
    public void testConvolutionModeOutput(){

        //Idea: with convolution mode == Truncate, input size shouldn't matter (within the bounds of truncated edge),
        // and edge data shouldn't affect the output

        //Use: 9x9, kernel 3, stride 3, padding 0
        // Should get same output for 10x10 and 11x11...

        Nd4j.getRandom().setSeed(12345);
        int[] minibatches = {1,3};
        int[] inDepths = {1,3};
        int[] inSizes = {9,10,11};

        for(boolean isSubsampling : new boolean[]{false, true}) {
            for (int minibatch : minibatches) {
                for (int inDepth : inDepths) {

                    INDArray origData = Nd4j.rand(new int[]{minibatch, inDepth, 9, 9});

                    for (int inSize : inSizes) {

                        for (ConvolutionMode cm : new ConvolutionMode[]{ConvolutionMode.Strict, ConvolutionMode.Truncate}) {

                            INDArray inputData = Nd4j.rand(new int[]{minibatch, inDepth, inSize, inSize});
                            inputData.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 9), NDArrayIndex.interval(0, 9))
                                    .assign(origData);

                            Layer layer;
                            if(isSubsampling){
                                layer = new SubsamplingLayer.Builder().kernelSize(3,3).stride(3,3).padding(0,0).build();
                            } else {
                                layer = new ConvolutionLayer.Builder().kernelSize(3, 3).stride(3, 3).padding(0, 0).nOut(3).build();
                            }

                            MultiLayerNetwork net = null;
                            try {
                                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                                        .weightInit(WeightInit.XAVIER)
                                        .convolutionMode(cm)
                                        .list()
                                        .layer(0, layer)
                                        .layer(1, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT).nOut(3).build())
                                        .setInputType(InputType.convolutional(inSize, inSize, inDepth))
                                        .build();

                                net = new MultiLayerNetwork(conf);
                                net.init();
                                if (inSize > 9 && cm == ConvolutionMode.Strict) {
                                    fail("Expected exception");
                                }
                            } catch (DL4JException e) {
                                if (inSize == 9 || cm != ConvolutionMode.Strict) {
                                    e.printStackTrace();
                                    fail("Unexpected exception");
                                }
                                continue;   //Expected exception
                            } catch (Exception e) {
                                e.printStackTrace();
                                fail("Unexpected exception");
                            }

                            INDArray out = net.output(origData);
                            INDArray out2 = net.output(inputData);

                            assertEquals(out, out2);
                        }
                    }
                }
            }
        }
    }

    @Test
    public void testConvolutionModeCompGraph(){

        //Idea: with convolution mode == Truncate, input size shouldn't matter (within the bounds of truncated edge),
        // and edge data shouldn't affect the output

        //Use: 9x9, kernel 3, stride 3, padding 0
        // Should get same output for 10x10 and 11x11...

        Nd4j.getRandom().setSeed(12345);
        int[] minibatches = {1,3};
        int[] inDepths = {1,3};
        int[] inSizes = {9,10,11};

        for(boolean isSubsampling : new boolean[]{false, true}) {
            for (int minibatch : minibatches) {
                for (int inDepth : inDepths) {

                    INDArray origData = Nd4j.rand(new int[]{minibatch, inDepth, 9, 9});

                    for (int inSize : inSizes) {

                        for (ConvolutionMode cm : new ConvolutionMode[]{ConvolutionMode.Strict, ConvolutionMode.Truncate}) {

                            INDArray inputData = Nd4j.rand(new int[]{minibatch, inDepth, inSize, inSize});
                            inputData.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(0, 9), NDArrayIndex.interval(0, 9))
                                    .assign(origData);

                            Layer layer;
                            if(isSubsampling){
                                layer = new SubsamplingLayer.Builder().kernelSize(3,3).stride(3,3).padding(0,0).build();
                            } else {
                                layer = new ConvolutionLayer.Builder().kernelSize(3, 3).stride(3, 3).padding(0, 0).nOut(3).build();
                            }

                            ComputationGraph net = null;
                            try {
                                ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                                        .weightInit(WeightInit.XAVIER)
                                        .convolutionMode(cm)
                                        .graphBuilder()
                                        .addInputs("in")
                                        .addLayer("0", layer, "in")
                                        .addLayer("1", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT).nOut(3).build(), "0")
                                        .setOutputs("1")
                                        .setInputTypes(InputType.convolutional(inSize, inSize, inDepth))
                                        .build();

                                net = new ComputationGraph(conf);
                                net.init();
                                if (inSize > 9 && cm == ConvolutionMode.Strict) {
                                    fail("Expected exception");
                                }
                            } catch (DL4JException e) {
                                if (inSize == 9 || cm != ConvolutionMode.Strict) {
                                    e.printStackTrace();
                                    fail("Unexpected exception");
                                }
                                continue;   //Expected exception
                            } catch (Exception e) {
                                e.printStackTrace();
                                fail("Unexpected exception");
                            }

                            INDArray out = net.outputSingle(origData);
                            INDArray out2 = net.outputSingle(inputData);

                            assertEquals(out, out2);
                        }
                    }
                }
            }
        }
    }

    @Test
    public void testGlobalLocalConfig(){

        for (ConvolutionMode cm : new ConvolutionMode[]{ConvolutionMode.Strict, ConvolutionMode.Truncate}) {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .weightInit(WeightInit.XAVIER)
                    .convolutionMode(cm)
                    .list()
                    .layer(0, new ConvolutionLayer.Builder().kernelSize(3, 3).stride(3, 3).padding(0, 0).nIn(3).nOut(3).build())
                    .layer(1, new ConvolutionLayer.Builder().convolutionMode(ConvolutionMode.Strict).kernelSize(3, 3).stride(3, 3).padding(0, 0).nIn(3).nOut(3).build())
                    .layer(2, new ConvolutionLayer.Builder().convolutionMode(ConvolutionMode.Truncate).kernelSize(3, 3).stride(3, 3).padding(0, 0).nIn(3).nOut(3).build())
                    .layer(3, new SubsamplingLayer.Builder().kernelSize(3, 3).stride(3, 3).padding(0, 0).build())
                    .layer(4, new SubsamplingLayer.Builder().convolutionMode(ConvolutionMode.Strict).kernelSize(3, 3).stride(3, 3).padding(0, 0).build())
                    .layer(5, new SubsamplingLayer.Builder().convolutionMode(ConvolutionMode.Truncate).kernelSize(3, 3).stride(3, 3).padding(0, 0).build())
                    .layer(6, new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT).nOut(3).build())
                    .build();

            assertEquals(cm, ((ConvolutionLayer)conf.getConf(0).getLayer()).getConvolutionMode());
            assertEquals(ConvolutionMode.Strict, ((ConvolutionLayer)conf.getConf(1).getLayer()).getConvolutionMode());
            assertEquals(ConvolutionMode.Truncate, ((ConvolutionLayer)conf.getConf(2).getLayer()).getConvolutionMode());

            assertEquals(cm, ((SubsamplingLayer)conf.getConf(3).getLayer()).getConvolutionMode());
            assertEquals(ConvolutionMode.Strict, ((SubsamplingLayer)conf.getConf(4).getLayer()).getConvolutionMode());
            assertEquals(ConvolutionMode.Truncate, ((SubsamplingLayer)conf.getConf(5).getLayer()).getConvolutionMode());
        }

    }

    @Test
    public void testGlobalLocalConfigCompGraph(){

        for (ConvolutionMode cm : new ConvolutionMode[]{ConvolutionMode.Strict, ConvolutionMode.Truncate}) {
            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                    .weightInit(WeightInit.XAVIER)
                    .convolutionMode(cm)
                    .graphBuilder()
                    .addInputs("in")
                    .addLayer("0", new ConvolutionLayer.Builder().kernelSize(3, 3).stride(3, 3).padding(0, 0).nIn(3).nOut(3).build(), "in")
                    .addLayer("1", new ConvolutionLayer.Builder().convolutionMode(ConvolutionMode.Strict).kernelSize(3, 3).stride(3, 3).padding(0, 0).nIn(3).nOut(3).build(), "0")
                    .addLayer("2", new ConvolutionLayer.Builder().convolutionMode(ConvolutionMode.Truncate).kernelSize(3, 3).stride(3, 3).padding(0, 0).nIn(3).nOut(3).build(), "1")
                    .addLayer("3", new SubsamplingLayer.Builder().kernelSize(3, 3).stride(3, 3).padding(0, 0).build(), "2")
                    .addLayer("4", new SubsamplingLayer.Builder().convolutionMode(ConvolutionMode.Strict).kernelSize(3, 3).stride(3, 3).padding(0, 0).build(), "3")
                    .addLayer("5", new SubsamplingLayer.Builder().convolutionMode(ConvolutionMode.Truncate).kernelSize(3, 3).stride(3, 3).padding(0, 0).build(), "4")
                    .addLayer("6", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MCXENT).nOut(3).build(), "5")
                    .setOutputs("6")
                    .build();

            assertEquals(cm, ((ConvolutionLayer)((LayerVertex)conf.getVertices().get("0")).getLayerConf().getLayer()).getConvolutionMode());
            assertEquals(ConvolutionMode.Strict, ((ConvolutionLayer)((LayerVertex)conf.getVertices().get("1")).getLayerConf().getLayer()).getConvolutionMode());
            assertEquals(ConvolutionMode.Truncate, ((ConvolutionLayer)((LayerVertex)conf.getVertices().get("2")).getLayerConf().getLayer()).getConvolutionMode());

            assertEquals(cm, ((SubsamplingLayer)((LayerVertex)conf.getVertices().get("3")).getLayerConf().getLayer()).getConvolutionMode());
            assertEquals(ConvolutionMode.Strict, ((SubsamplingLayer)((LayerVertex)conf.getVertices().get("4")).getLayerConf().getLayer()).getConvolutionMode());
            assertEquals(ConvolutionMode.Truncate, ((SubsamplingLayer)((LayerVertex)conf.getVertices().get("5")).getLayerConf().getLayer()).getConvolutionMode());
        }

    }
}
