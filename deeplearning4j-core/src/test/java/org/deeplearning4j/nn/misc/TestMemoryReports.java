package org.deeplearning4j.nn.misc;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.*;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryType;
import org.deeplearning4j.nn.conf.memory.MemoryUseMode;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 14/07/2017.
 */
public class TestMemoryReports extends BaseDL4JTest {

    public static List<Pair<? extends Layer, InputType>> getTestLayers() {
        List<Pair<? extends Layer, InputType>> l = new ArrayList<>();
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

    public static List<Pair<? extends GraphVertex, InputType[]>> getTestVertices() {

        List<Pair<? extends GraphVertex, InputType[]>> out = new ArrayList<>();
        out.add(new Pair<>(new ElementWiseVertex(ElementWiseVertex.Op.Add),
                        new InputType[] {InputType.feedForward(10), InputType.feedForward(10)}));
        out.add(new Pair<>(new ElementWiseVertex(ElementWiseVertex.Op.Add),
                        new InputType[] {InputType.recurrent(10, 10), InputType.recurrent(10, 10)}));
        out.add(new Pair<>(new L2NormalizeVertex(), new InputType[] {InputType.feedForward(10)}));
        out.add(new Pair<>(new L2Vertex(), new InputType[] {InputType.recurrent(10, 10), InputType.recurrent(10, 10)}));
        out.add(new Pair<>(new MergeVertex(),
                        new InputType[] {InputType.recurrent(10, 10), InputType.recurrent(10, 10)}));
        out.add(new Pair<>(new PreprocessorVertex(new FeedForwardToCnnPreProcessor(1, 10, 1)),
                        new InputType[] {InputType.convolutional(1, 10, 1)}));
        out.add(new Pair<>(new ScaleVertex(1.0), new InputType[] {InputType.recurrent(10, 10)}));
        out.add(new Pair<>(new ShiftVertex(1.0), new InputType[] {InputType.recurrent(10, 10)}));
        out.add(new Pair<>(new StackVertex(),
                        new InputType[] {InputType.recurrent(10, 10), InputType.recurrent(10, 10)}));
        out.add(new Pair<>(new UnstackVertex(0, 2), new InputType[] {InputType.recurrent(10, 10)}));

        out.add(new Pair<>(new DuplicateToTimeSeriesVertex("0"),
                        new InputType[] {InputType.recurrent(10, 10), InputType.feedForward(10)}));
        out.add(new Pair<>(new LastTimeStepVertex("0"), new InputType[] {InputType.recurrent(10, 10)}));

        return out;
    }

    @Test
    public void testMemoryReportSimple() {

        List<Pair<? extends Layer, InputType>> l = getTestLayers();


        for (Pair<? extends Layer, InputType> p : l) {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list().layer(0, p.getFirst().clone())
                            .layer(1, p.getFirst().clone()).build();

            MemoryReport mr = conf.getMemoryReport(p.getSecond());
            //            System.out.println(mr.toString());
            //            System.out.println("\n\n");

            //Test to/from JSON + YAML
            String json = mr.toJson();
            String yaml = mr.toYaml();

            MemoryReport fromJson = MemoryReport.fromJson(json);
            MemoryReport fromYaml = MemoryReport.fromYaml(yaml);

            assertEquals(mr, fromJson);
            assertEquals(mr, fromYaml);
        }
    }


    @Test
    public void testMemoryReportSimpleCG() {

        List<Pair<? extends Layer, InputType>> l = getTestLayers();


        for (Pair<? extends Layer, InputType> p : l) {

            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs("in")
                            .addLayer("0", p.getFirst().clone(), "in").addLayer("1", p.getFirst().clone(), "0")
                            .setOutputs("1").build();

            MemoryReport mr = conf.getMemoryReport(p.getSecond());
            //            System.out.println(mr.toString());
            //            System.out.println("\n\n");

            //Test to/from JSON + YAML
            String json = mr.toJson();
            String yaml = mr.toYaml();

            MemoryReport fromJson = MemoryReport.fromJson(json);
            MemoryReport fromYaml = MemoryReport.fromYaml(yaml);

            assertEquals(mr, fromJson);
            assertEquals(mr, fromYaml);
        }
    }

    @Test
    public void testMemoryReportsVerticesCG() {
        List<Pair<? extends GraphVertex, InputType[]>> l = getTestVertices();

        for (Pair<? extends GraphVertex, InputType[]> p : l) {
            List<String> inputs = new ArrayList<>();
            for (int i = 0; i < p.getSecond().length; i++) {
                inputs.add(String.valueOf(i));
            }

            String[] layerInputs = inputs.toArray(new String[inputs.size()]);
            if (p.getFirst() instanceof DuplicateToTimeSeriesVertex) {
                layerInputs = new String[] {"1"};
            }

            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().graphBuilder().addInputs(inputs)
                            .allowDisconnected(true)
                            .addVertex("gv", p.getFirst(), layerInputs).setOutputs("gv").build();

            MemoryReport mr = conf.getMemoryReport(p.getSecond());
            //            System.out.println(mr.toString());
            //            System.out.println("\n\n");

            //Test to/from JSON + YAML
            String json = mr.toJson();
            String yaml = mr.toYaml();

            MemoryReport fromJson = MemoryReport.fromJson(json);
            MemoryReport fromYaml = MemoryReport.fromYaml(yaml);

            assertEquals(mr, fromJson);
            assertEquals(mr, fromYaml);
        }
    }


    @Test
    public void testInferInputType() {
        List<Pair<INDArray[], InputType[]>> l = new ArrayList<>();
        l.add(new Pair<>(new INDArray[] {Nd4j.create(10, 8)}, new InputType[] {InputType.feedForward(8)}));
        l.add(new Pair<>(new INDArray[] {Nd4j.create(10, 8), Nd4j.create(10, 20)},
                        new InputType[] {InputType.feedForward(8), InputType.feedForward(20)}));
        l.add(new Pair<>(new INDArray[] {Nd4j.create(10, 8, 7)}, new InputType[] {InputType.recurrent(8, 7)}));
        l.add(new Pair<>(new INDArray[] {Nd4j.create(10, 8, 7), Nd4j.create(10, 20, 6)},
                        new InputType[] {InputType.recurrent(8, 7), InputType.recurrent(20, 6)}));

        //Activations order: [m,d,h,w]
        l.add(new Pair<>(new INDArray[] {Nd4j.create(10, 8, 7, 6)},
                        new InputType[] {InputType.convolutional(7, 6, 8)}));
        l.add(new Pair<>(new INDArray[] {Nd4j.create(10, 8, 7, 6), Nd4j.create(10, 4, 3, 2),},
                        new InputType[] {InputType.convolutional(7, 6, 8), InputType.convolutional(3, 2, 4)}));

        for (Pair<INDArray[], InputType[]> p : l) {
            InputType[] act = InputType.inferInputTypes(p.getFirst());

            assertArrayEquals(p.getSecond(), act);
        }
    }


    @Test
    public void validateSimple() {

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().list()
                        .layer(0, new DenseLayer.Builder().nIn(10).nOut(20).build())
                        .layer(1, new DenseLayer.Builder().nIn(20).nOut(27).build()).build();

        MemoryReport mr = conf.getMemoryReport(InputType.feedForward(10));

        int numParams = (10 * 20 + 20) + (20 * 27 + 27); //787 -> 3148 bytes
        int actSize = 20 + 27; //47 -> 188 bytes
        int total15Minibatch = numParams + 15 * actSize;

        //Fixed: should be just params
        long fixedBytes = mr.getTotalMemoryBytes(0, MemoryUseMode.INFERENCE, CacheMode.NONE, DataBuffer.Type.FLOAT);
        long varBytes = mr.getTotalMemoryBytes(1, MemoryUseMode.INFERENCE, CacheMode.NONE, DataBuffer.Type.FLOAT)
                        - fixedBytes;

        assertEquals(numParams * 4, fixedBytes);
        assertEquals(actSize * 4, varBytes);

        long minibatch15 = mr.getTotalMemoryBytes(15, MemoryUseMode.INFERENCE, CacheMode.NONE, DataBuffer.Type.FLOAT);
        assertEquals(total15Minibatch * 4, minibatch15);

        //        System.out.println(fixedBytes + "\t" + varBytes);
        //        System.out.println(mr.toString());

        assertEquals(actSize * 4, mr.getMemoryBytes(MemoryType.ACTIVATIONS, 1, MemoryUseMode.TRAINING, CacheMode.NONE,
                        DataBuffer.Type.FLOAT));
        assertEquals(actSize * 4, mr.getMemoryBytes(MemoryType.ACTIVATIONS, 1, MemoryUseMode.INFERENCE, CacheMode.NONE,
                        DataBuffer.Type.FLOAT));

        int inputActSize = 10 + 20;
        assertEquals(inputActSize * 4, mr.getMemoryBytes(MemoryType.ACTIVATION_GRADIENTS, 1, MemoryUseMode.TRAINING,
                        CacheMode.NONE, DataBuffer.Type.FLOAT));
        assertEquals(0, mr.getMemoryBytes(MemoryType.ACTIVATION_GRADIENTS, 1, MemoryUseMode.INFERENCE, CacheMode.NONE,
                        DataBuffer.Type.FLOAT));

        //Variable working memory - due to preout during backprop. But not it's the MAX value, as it can be GC'd or workspaced
        int workingMemVariable = 27;
        assertEquals(workingMemVariable * 4, mr.getMemoryBytes(MemoryType.WORKING_MEMORY_VARIABLE, 1,
                        MemoryUseMode.TRAINING, CacheMode.NONE, DataBuffer.Type.FLOAT));
        assertEquals(0, mr.getMemoryBytes(MemoryType.WORKING_MEMORY_VARIABLE, 1, MemoryUseMode.INFERENCE,
                        CacheMode.NONE, DataBuffer.Type.FLOAT));
    }

    @Test
    public void testPreprocessors() throws Exception {
        //https://github.com/deeplearning4j/deeplearning4j/issues/4223
        File f = new ClassPathResource("4223/CompGraphConfig.json").getTempFileFromArchive();
        String s = FileUtils.readFileToString(f, Charset.defaultCharset());

        ComputationGraphConfiguration conf = ComputationGraphConfiguration.fromJson(s);

        conf.getMemoryReport(InputType.convolutional(17,19,19));
    }
}
