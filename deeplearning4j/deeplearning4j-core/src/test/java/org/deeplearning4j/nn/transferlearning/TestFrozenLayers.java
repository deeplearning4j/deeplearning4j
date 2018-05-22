package org.deeplearning4j.nn.transferlearning;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.Assert.*;

public class TestFrozenLayers extends BaseDL4JTest {

    @Test
    public void testFrozenMLN(){
        MultiLayerNetwork orig = getOriginalNet(12345);


        for(double l1 : new double[]{0.0, 0.3}){
            for( double l2 : new double[]{0.0, 0.4}){
                System.out.println("--------------------");
                String msg = "l1=" + l1 + ", l2=" + l2;

                FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
                        .updater(new Sgd(0.5))
                        .l1(l1)
                        .l2(l2)
                        .build();

                MultiLayerNetwork transfer = new TransferLearning.Builder(orig)
                        .fineTuneConfiguration(ftc)
                        .setFeatureExtractor(4)
                        .removeOutputLayer()
                        .addLayer(new OutputLayer.Builder().nIn(64).nOut(10).lossFunction(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR).build())
                        .build();

                assertEquals(6, transfer.getnLayers());
                for( int i=0; i<5; i++ ){
                    assertTrue( transfer.getLayer(i) instanceof FrozenLayer);
                }

                Map<String,INDArray> paramsBefore = new LinkedHashMap<>();
                for(Map.Entry<String,INDArray> entry : transfer.paramTable().entrySet()){
                    paramsBefore.put(entry.getKey(), entry.getValue().dup());
                }

                for( int i=0; i<20; i++ ){
                    INDArray f = Nd4j.rand(new int[]{16,1,28,28});
                    INDArray l = Nd4j.rand(new int[]{16,10});
                    transfer.fit(f,l);
                }

                for(Map.Entry<String,INDArray> entry : transfer.paramTable().entrySet()){
                    String s = msg + " - " + entry.getKey();
                    if(entry.getKey().startsWith("5_")){
                        //Non-frozen layer
                        assertNotEquals(s, paramsBefore.get(entry.getKey()), entry.getValue());
                    } else {
                        assertEquals(s, paramsBefore.get(entry.getKey()), entry.getValue());
                    }
                }
            }
        }
    }

    @Test
    public void testFrozenCG(){
        ComputationGraph orig = getOriginalGraph(12345);


        for(double l1 : new double[]{0.0, 0.3}){
            for( double l2 : new double[]{0.0, 0.4}){
                String msg = "l1=" + l1 + ", l2=" + l2;

                FineTuneConfiguration ftc = new FineTuneConfiguration.Builder()
                        .updater(new Sgd(0.5))
                        .l1(l1)
                        .l2(l2)
                        .build();

                ComputationGraph transfer = new TransferLearning.GraphBuilder(orig)
                        .fineTuneConfiguration(ftc)
                        .setFeatureExtractor("4")
                        .removeVertexAndConnections("5")
                        .addLayer("5", new OutputLayer.Builder().nIn(64).nOut(10).lossFunction(LossFunctions.LossFunction.MEAN_ABSOLUTE_ERROR).build(), "4")
                        .setOutputs("5")
                        .build();

                assertEquals(6, transfer.getNumLayers());
                for( int i=0; i<5; i++ ){
                    assertTrue( transfer.getLayer(i) instanceof FrozenLayer);
                }

                Map<String,INDArray> paramsBefore = new LinkedHashMap<>();
                for(Map.Entry<String,INDArray> entry : transfer.paramTable().entrySet()){
                    paramsBefore.put(entry.getKey(), entry.getValue().dup());
                }

                for( int i=0; i<20; i++ ){
                    INDArray f = Nd4j.rand(new int[]{16,1,28,28});
                    INDArray l = Nd4j.rand(new int[]{16,10});
                    transfer.fit(new INDArray[]{f},new INDArray[]{l});
                }

                for(Map.Entry<String,INDArray> entry : transfer.paramTable().entrySet()){
                    String s = msg + " - " + entry.getKey();
                    if(entry.getKey().startsWith("5_")){
                        //Non-frozen layer
                        assertNotEquals(s, paramsBefore.get(entry.getKey()), entry.getValue());
                    } else {
                        assertEquals(s, paramsBefore.get(entry.getKey()), entry.getValue());
                    }
                }
            }
        }
    }

    public static MultiLayerNetwork getOriginalNet(int seed){
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.TANH)
                .convolutionMode(ConvolutionMode.Same)
                .updater(new Sgd(0.3))
                .list()
                .layer(new ConvolutionLayer.Builder().nOut(3).kernelSize(2,2).stride(1,1).build())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(1,1).build())
                .layer(new ConvolutionLayer.Builder().nIn(3).nOut(3).kernelSize(2,2).stride(1,1).build())
                .layer(new DenseLayer.Builder().nOut(64).build())
                .layer(new DenseLayer.Builder().nIn(64).nOut(64).build())
                .layer(new OutputLayer.Builder().nIn(64).nOut(10).lossFunction(LossFunctions.LossFunction.MSE).build())
                .setInputType(InputType.convolutionalFlat(28,28,1))
                .build();


        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        return net;
    }

    public static ComputationGraph getOriginalGraph(int seed){
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.TANH)
                .convolutionMode(ConvolutionMode.Same)
                .updater(new Sgd(0.3))
                .graphBuilder()
                .addInputs("in")
                .layer("0", new ConvolutionLayer.Builder().nOut(3).kernelSize(2,2).stride(1,1).build(), "in")
                .layer("1", new SubsamplingLayer.Builder().kernelSize(2,2).stride(1,1).build(), "0")
                .layer("2", new ConvolutionLayer.Builder().nIn(3).nOut(3).kernelSize(2,2).stride(1,1).build(), "1")
                .layer("3", new DenseLayer.Builder().nOut(64).build(), "2")
                .layer("4", new DenseLayer.Builder().nIn(64).nOut(64).build(), "3")
                .layer("5", new OutputLayer.Builder().nIn(64).nOut(10).lossFunction(LossFunctions.LossFunction.MSE).build(), "4")
                .setOutputs("5")
                .setInputTypes(InputType.convolutionalFlat(28,28,1))
                .build();


        ComputationGraph net = new ComputationGraph(conf);
        net.init();
        return net;
    }

}
