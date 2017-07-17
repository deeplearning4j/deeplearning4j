package org.deeplearning4j.nn.misc;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.*;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.fail;

/**
 * Created by Alex on 14/07/2017.
 */
public class TestMemoryReports {

    public static List<Pair<? extends Layer,InputType>> getTestLayers(){
        List<Pair<? extends Layer,InputType>> l = new ArrayList<>();
        l.add(new Pair<>(new ActivationLayer.Builder().activation(Activation.TANH).build(), InputType.feedForward(20)));
        l.add(new Pair<>(new DenseLayer.Builder().nIn(20).nOut(20).build(), InputType.feedForward(20)));
        l.add(new Pair<>(new DropoutLayer.Builder().nIn(20).nOut(20).build(), InputType.feedForward(20)));
        l.add(new Pair<>(new EmbeddingLayer.Builder().nIn(1).nOut(20).build(), InputType.feedForward(20)));
        l.add(new Pair<>(new OutputLayer.Builder().nIn(20).nOut(20).build(), InputType.feedForward(20)));
        l.add(new Pair<>(new LossLayer.Builder().build(), InputType.feedForward(20)));

        //RNN layers:
        l.add(new Pair<>(new GravesLSTM.Builder().nIn(20).nOut(20).build(), InputType.recurrent(20, 30)));
        l.add(new Pair<>(new LSTM.Builder().nIn(20).nOut(20).build(), InputType.recurrent(20, 30)));
        l.add(new Pair<>(new GravesBidirectionalLSTM.Builder().nIn(20).nOut(20).build(), InputType.recurrent(20, 30)));
        l.add(new Pair<>(new RnnOutputLayer.Builder().nIn(20).nOut(20).build(), InputType.recurrent(20, 30)));

        return l;
    }

    public static List<Pair<? extends GraphVertex, InputType[]>> getTestVertices(){

        List<Pair<? extends GraphVertex, InputType[]>> out = new ArrayList<>();
        out.add(new Pair<>(new ElementWiseVertex(ElementWiseVertex.Op.Add), new InputType[]{InputType.feedForward(10), InputType.feedForward(10)}));
        out.add(new Pair<>(new ElementWiseVertex(ElementWiseVertex.Op.Add), new InputType[]{InputType.recurrent(10,10), InputType.recurrent(10,10)}));
        out.add(new Pair<>(new L2NormalizeVertex(), new InputType[]{InputType.feedForward(10)}));
        out.add(new Pair<>(new L2Vertex(), new InputType[]{InputType.recurrent(10,10), InputType.recurrent(10,10)}));
        out.add(new Pair<>(new MergeVertex(), new InputType[]{InputType.recurrent(10,10), InputType.recurrent(10,10)}));
        out.add(new Pair<>(new PreprocessorVertex(new FeedForwardToCnnPreProcessor(1,10,1)), new InputType[]{InputType.convolutional(1,10,1)}));
        out.add(new Pair<>(new ScaleVertex(1.0), new InputType[]{InputType.recurrent(10,10)}));
        out.add(new Pair<>(new ShiftVertex(1.0), new InputType[]{InputType.recurrent(10,10)}));
        out.add(new Pair<>(new StackVertex(), new InputType[]{InputType.recurrent(10,10), InputType.recurrent(10,10)}));
        out.add(new Pair<>(new UnstackVertex(0,2), new InputType[]{InputType.recurrent(10,10)}));

        out.add(new Pair<>(new DuplicateToTimeSeriesVertex("0"), new InputType[]{InputType.recurrent(10,10), InputType.feedForward(10)}));
        out.add(new Pair<>(new LastTimeStepVertex("0"), new InputType[]{InputType.recurrent(10,10)}));

        return out;
    }

    @Test
    public void testMemoryReportSimple(){

        List<Pair<? extends Layer,InputType>> l = getTestLayers();


        for(Pair<? extends Layer,InputType> p : l){

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .list()
                    .layer(0, p.getFirst().clone())
                    .layer(1, p.getFirst().clone())
                    .build();

            MemoryReport mr = conf.getMemoryReport(p.getSecond());
            System.out.println(mr.toString());

            System.out.println("\n\n");
        }
    }


    @Test
    public void testMemoryReportSimpleCG(){

        List<Pair<? extends Layer,InputType>> l = getTestLayers();


        for(Pair<? extends Layer,InputType> p : l){

            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                    .graphBuilder()
                    .addInputs("in")
                    .addLayer("0", p.getFirst().clone(), "in")
                    .addLayer("1", p.getFirst().clone(), "0")
                    .setOutputs("1")
                    .build();

            MemoryReport mr = conf.getMemoryReport(p.getSecond());
            System.out.println(mr.toString());

            System.out.println("\n\n");
        }
    }

    @Test
    public void testMemoryReportsVerticesCG(){
        List<Pair<? extends GraphVertex, InputType[]>> l = getTestVertices();

        for(Pair<? extends GraphVertex,InputType[]> p : l){
            List<String> inputs = new ArrayList<>();
            for( int i=0; i<p.getSecond().length; i++ ){
                inputs.add(String.valueOf(i));
            }

            String[] layerInputs = inputs.toArray(new String[inputs.size()]);
            if(p.getFirst() instanceof DuplicateToTimeSeriesVertex){
                layerInputs = new String[]{"1"};
            }

            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                    .graphBuilder()
                    .addInputs(inputs)
                    .addVertex("gv", p.getFirst(), layerInputs)
                    .setOutputs("gv")
                    .build();

            MemoryReport mr = conf.getMemoryReport(p.getSecond());
            System.out.println(mr.toString());

            System.out.println("\n\n");
        }


    }
}
