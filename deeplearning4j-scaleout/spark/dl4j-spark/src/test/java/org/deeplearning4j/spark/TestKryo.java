package org.deeplearning4j.spark;

import org.apache.spark.serializer.SerializerInstance;
import org.deeplearning4j.eval.*;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.graph.*;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.preprocessor.CnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import scala.collection.JavaConversions;

import java.nio.ByteBuffer;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 04/07/2017.
 */
public class TestKryo extends BaseSparkKryoTest {

    private <T> void testSerialization(T in, SerializerInstance si) {
        ByteBuffer bb = si.serialize(in, null);
        T deserialized = (T)si.deserialize(bb, null);

        boolean equals = in.equals(deserialized);
        assertTrue(in.getClass() + "\t" + in.toString(), equals);
    }

    @Test
    public void testSerializationConfigurations() {

        SerializerInstance si = sc.env().serializer().newInstance();

        //Check network configurations:
        Map<Integer, Double> m = new HashMap<>();
        m.put(0, 0.5);
        m.put(10, 0.1);
        MultiLayerConfiguration mlc = new NeuralNetConfiguration.Builder()
                        .updater(new Nadam(new MapSchedule(ScheduleType.ITERATION,m))).list().layer(0, new OutputLayer.Builder().nIn(10).nOut(10).build())
                        .build();

        testSerialization(mlc, si);


        ComputationGraphConfiguration cgc = new NeuralNetConfiguration.Builder().weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(-1, 1))
                        .updater(new Adam(new MapSchedule(ScheduleType.ITERATION,m)))
                        .graphBuilder()
                        .addInputs("in").addLayer("out", new OutputLayer.Builder().nIn(10).nOut(10).build(), "in")
                        .setOutputs("out").build();

        testSerialization(cgc, si);


        //Check main layers:
        Layer[] layers = new Layer[] {new OutputLayer.Builder().nIn(10).nOut(10).build(),
                        new RnnOutputLayer.Builder().nIn(10).nOut(10).build(), new LossLayer.Builder().build(),
                        new CenterLossOutputLayer.Builder().nIn(10).nOut(10).build(),
                        new DenseLayer.Builder().nIn(10).nOut(10).build(),
                        new ConvolutionLayer.Builder().nIn(10).nOut(10).build(), new SubsamplingLayer.Builder().build(),
                        new Convolution1DLayer.Builder(2, 2).nIn(10).nOut(10).build(),
                        new ActivationLayer.Builder().activation(Activation.TANH).build(),
                        new GlobalPoolingLayer.Builder().build(), new GravesLSTM.Builder().nIn(10).nOut(10).build(),
                        new LSTM.Builder().nIn(10).nOut(10).build(), new DropoutLayer.Builder(0.5).build(),
                        new BatchNormalization.Builder().build(), new LocalResponseNormalization.Builder().build()};

        for (Layer l : layers) {
            testSerialization(l, si);
        }

        //Check graph vertices
        GraphVertex[] vertices = new GraphVertex[] {new ElementWiseVertex(ElementWiseVertex.Op.Add),
                        new L2NormalizeVertex(), new LayerVertex(null, null), new MergeVertex(), new PoolHelperVertex(),
                        new PreprocessorVertex(new CnnToFeedForwardPreProcessor(28, 28, 1)),
                        new ReshapeVertex(new int[] {1, 1}), new ScaleVertex(1.0), new ShiftVertex(1.0),
                        new SubsetVertex(1, 1), new UnstackVertex(0, 2), new DuplicateToTimeSeriesVertex("in1"),
                        new LastTimeStepVertex("in1")};

        for (GraphVertex gv : vertices) {
            testSerialization(gv, si);
        }
    }

    @Test
    public void testSerializationEvaluation() {

        Evaluation e = new Evaluation();
        e.eval(Nd4j.create(new double[] {1, 0, 0}), Nd4j.create(new double[] {0.2, 0.5, 0.3}));

        EvaluationBinary eb = new EvaluationBinary();
        eb.eval(Nd4j.create(new double[] {1, 0, 0}), Nd4j.create(new double[] {0.2, 0.6, 0.3}));

        ROC roc = new ROC(30);
        roc.eval(Nd4j.create(new double[] {1}), Nd4j.create(new double[] {0.2}));
        ROC roc2 = new ROC();
        roc2.eval(Nd4j.create(new double[] {1}), Nd4j.create(new double[] {0.2}));

        ROCMultiClass rocM = new ROCMultiClass(30);
        rocM.eval(Nd4j.create(new double[] {1, 0, 0}), Nd4j.create(new double[] {0.2, 0.5, 0.3}));
        ROCMultiClass rocM2 = new ROCMultiClass();
        rocM2.eval(Nd4j.create(new double[] {1, 0, 0}), Nd4j.create(new double[] {0.2, 0.5, 0.3}));

        ROCBinary rocB = new ROCBinary(30);
        rocB.eval(Nd4j.create(new double[] {1, 0, 0}), Nd4j.create(new double[] {0.2, 0.6, 0.3}));

        ROCBinary rocB2 = new ROCBinary();
        rocB2.eval(Nd4j.create(new double[] {1, 0, 0}), Nd4j.create(new double[] {0.2, 0.6, 0.3}));

        RegressionEvaluation re = new RegressionEvaluation();
        re.eval(Nd4j.rand(1, 5), Nd4j.rand(1, 5));

        IEvaluation[] evaluations = new IEvaluation[] {new Evaluation(), e, new EvaluationBinary(), eb, new ROC(), roc,
                        roc2, new ROCMultiClass(), rocM, rocM2, new ROCBinary(), rocB, rocB2,
                        new RegressionEvaluation(), re};

        SerializerInstance si = sc.env().serializer().newInstance();

        for (IEvaluation ie : evaluations) {
            //System.out.println(ie.getClass());
            testSerialization(ie, si);
        }
    }

    @Test
    public void testScalaCollections() {
        //Scala collections should already work with Spark + kryo; some very basic tests to check this is still the case
        SerializerInstance si = sc.env().serializer().newInstance();

        scala.collection.immutable.Map<Integer, String> emptyImmutableMap =
                        scala.collection.immutable.Map$.MODULE$.empty();
        testSerialization(emptyImmutableMap, si);

        Map<Integer, Double> m = new HashMap<>();
        m.put(0, 1.0);

        scala.collection.Map<Integer, Double> m2 = JavaConversions.mapAsScalaMap(m);
        testSerialization(m2, si);
    }

    @Test
    public void testJavaTypes() {

        Map<Object, Object> m = new HashMap<>();
        m.put("key", "value");

        SerializerInstance si = sc.env().serializer().newInstance();

        testSerialization(Collections.singletonMap("key", "value"), si);
        testSerialization(Collections.synchronizedMap(m), si);
        testSerialization(Collections.emptyMap(), si);
        testSerialization(new ConcurrentHashMap<>(m), si);
        testSerialization(Collections.unmodifiableMap(m), si);

        testSerialization(Arrays.asList("s"), si);
        testSerialization(Collections.singleton("s"), si);
        testSerialization(Collections.synchronizedList(Arrays.asList("s")), si);
        testSerialization(Collections.emptyList(), si);
        testSerialization(new CopyOnWriteArrayList<>(Arrays.asList("s")), si);
        testSerialization(Collections.unmodifiableList(Arrays.asList("s")), si);

        testSerialization(Collections.singleton("s"), si);
        testSerialization(Collections.synchronizedSet(new HashSet<>(Arrays.asList("s"))), si);
        testSerialization(Collections.emptySet(), si);
        testSerialization(Collections.unmodifiableSet(new HashSet<>(Arrays.asList("s"))), si);
    }
}
