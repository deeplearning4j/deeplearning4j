package org.deeplearning4j;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Debug {

    public static void main(String[] args){

        int lstmLayerSize = 16;
        int numLabelClasses = 10;
        int numInputs = 5;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
//            .trainingWorkspaceMode(WorkspaceMode.SINGLE)
//            .inferenceWorkspaceMode(WorkspaceMode.SINGLE)
                .trainingWorkspaceMode(WorkspaceMode.NONE)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
            .seed(123)    //Random number generator seed for improved repeatability. Optional.
            .updater(new AdaDelta())
            .weightInit(WeightInit.XAVIER)
            .graphBuilder()
            .addInputs("rr")
            .setInputTypes(InputType.recurrent(30))
            .addLayer("1", new GravesLSTM.Builder().activation(Activation.TANH).nIn(numInputs).nOut(lstmLayerSize).dropOut(0.9).build(), "rr")
            .addLayer("2", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                .activation(Activation.SOFTMAX).nOut(numLabelClasses).build(), "1")
            .pretrain(false).backprop(true)
            .setOutputs("2")
            .build();


        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        ComputationGraph updatedModel = new TransferLearning.GraphBuilder(net)
            .addVertex("laststepoutput", new LastTimeStepVertex("rr"), "2")
            .setOutputs("laststepoutput")
            .build();


        INDArray input = Nd4j.rand(new int[]{10, numInputs, 16});

        INDArray[] out = updatedModel.output(input);

        for( int i=0; i<out.length; i++ ){
//            System.out.println(i + "\t" + out[i].shapeInfoToString());
            System.out.println(i + "\t" + out[i]);
//            System.out.println(i + "\t" + out[i].detach());
        }

    }

}
