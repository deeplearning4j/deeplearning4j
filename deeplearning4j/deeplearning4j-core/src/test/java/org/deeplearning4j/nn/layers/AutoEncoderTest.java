package org.deeplearning4j.nn.layers;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.datasets.iterator.impl.SingletonMultiDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class AutoEncoderTest extends BaseDL4JTest {

    @Test
    public void sanityCheckIssue5662(){
        int mergeSize = 50;
        int encdecSize = 25;
        int in1Size = 20;
        int in2Size = 15;
        int hiddenSize = 10;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                .addInputs("in1", "in2")
                .addLayer("1", new DenseLayer.Builder().nOut(mergeSize).build(), "in1")
                .addLayer("2", new DenseLayer.Builder().nOut(mergeSize).build(), "in2")
                .addVertex("merge", new MergeVertex(), "1", "2")
                .addLayer("e",new AutoEncoder.Builder().nOut(encdecSize).corruptionLevel(0.2).build(),"merge")
                .addLayer("hidden",new AutoEncoder.Builder().nOut(hiddenSize).build(),"e")
                .addLayer("decoder",new AutoEncoder.Builder().nOut(encdecSize).corruptionLevel(0.2).build(),"hidden")
                .addLayer("L4", new DenseLayer.Builder().nOut(mergeSize).build(), "decoder")
                .addLayer("out1", new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nOut(in1Size).build(),"L4")
                .addLayer("out2",new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nOut(in2Size).build(),"L4")
                .setOutputs("out1","out2")
                .setInputTypes(InputType.feedForward(in1Size), InputType.feedForward(in2Size))
                .pretrain(true).backprop(true)
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        MultiDataSet mds = new org.nd4j.linalg.dataset.MultiDataSet(
                new INDArray[]{Nd4j.create(1, in1Size), Nd4j.create(1, in2Size)},
                new INDArray[]{Nd4j.create(1, in1Size), Nd4j.create(1, in2Size)});

        net.summary(InputType.feedForward(in1Size), InputType.feedForward(in2Size));
        net.fit(new SingletonMultiDataSetIterator(mds));
    }

}
