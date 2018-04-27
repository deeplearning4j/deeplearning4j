package org.deeplearning4j.nn.misc;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.misc.iter.WSTestDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.executioner.OpExecutioner;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

@Slf4j
public class WorkspaceTests extends BaseDL4JTest {

    @Before
    public void before() {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.SCOPE_PANIC);
    }

    @After
    public void after() {
        Nd4j.getExecutioner().setProfilingMode(OpExecutioner.ProfilingMode.DISABLED);
    }

    @Test
    public void checkScopesTestCGAS() throws Exception {
        ComputationGraph c = createNet();
        for (WorkspaceMode wm : new WorkspaceMode[]{WorkspaceMode.NONE, WorkspaceMode.ENABLED}) {
            log.info("Starting test: {}", wm);
            c.getConfiguration().setTrainingWorkspaceMode(wm);
            c.getConfiguration().setInferenceWorkspaceMode(wm);

            INDArray f = Nd4j.rand(new int[]{8, 1, 28, 28});
            INDArray l = Nd4j.rand(8, 10);
            c.setInputs(f);
            c.setLabels(l);

            c.computeGradientAndScore();
        }
    }


    @Test
    public void testWorkspaceIndependence() {
        //https://github.com/deeplearning4j/deeplearning4j/issues/4337
        int depthIn = 2;
        int depthOut = 2;
        int nOut = 2;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().weightInit(WeightInit.XAVIER)
                .convolutionMode(ConvolutionMode.Same).seed(12345L).list()
                .layer(0, new ConvolutionLayer.Builder().nIn(depthIn).nOut(depthOut).kernelSize(2, 2)
                        .stride(1, 1).activation(Activation.TANH).build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nOut(nOut).build())
                .setInputType(InputType.convolutional(5, 5, 2))
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf.clone());
        net.init();
        net.getLayerWiseConfigurations().setInferenceWorkspaceMode(WorkspaceMode.ENABLED);
        net.getLayerWiseConfigurations().setTrainingWorkspaceMode(WorkspaceMode.ENABLED);

        MultiLayerNetwork net2 = new MultiLayerNetwork(conf.clone());
        net2.init();
        net2.getLayerWiseConfigurations().setInferenceWorkspaceMode(WorkspaceMode.NONE);
        net2.getLayerWiseConfigurations().setTrainingWorkspaceMode(WorkspaceMode.NONE);

        INDArray in = Nd4j.rand(new int[]{1, 2, 5, 5});

        net.output(in);
        net2.output(in);    //Op [add_scalar] X argument uses leaked workspace pointer from workspace [LOOP_EXTERNAL]
    }

    public static ComputationGraph createNet() throws Exception {

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new ConvolutionLayer.Builder().nOut(3)
                        .kernelSize(2, 2).stride(2, 2).build(), "in")
                .addLayer("1", new ConvolutionLayer.Builder().nOut(3)
                        .kernelSize(2, 2).stride(2, 2).build(), "0")
                .addLayer("out", new OutputLayer.Builder().nOut(10)
                        .activation(Activation.TANH).lossFunction(LossFunctions.LossFunction.MSE)
                        .build(), "1")
                .setOutputs("out")
                .setInputTypes(InputType.convolutional(28, 28, 1))
                .build();

        ComputationGraph model = new ComputationGraph(conf);
        model.init();

        return model;
    }


    @Test
    public void testWithPreprocessorsCG() {
        //https://github.com/deeplearning4j/deeplearning4j/issues/4347
        //Cause for the above issue was layerVertex.setInput() applying the preprocessor, with the result
        // not being detached properly from the workspace...

        for (WorkspaceMode wm : WorkspaceMode.values()) {
            System.out.println(wm);
            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                    .trainingWorkspaceMode(wm)
                    .inferenceWorkspaceMode(wm)
                    .graphBuilder()
                    .addInputs("in")
                    .addLayer("e", new GravesLSTM.Builder().nIn(10).nOut(5).build(), new DupPreProcessor(), "in")
//                .addLayer("e", new GravesLSTM.Builder().nIn(10).nOut(5).build(), "in")    //Note that no preprocessor is OK
                    .addLayer("rnn", new GravesLSTM.Builder().nIn(5).nOut(8).build(), "e")
                    .addLayer("out", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .activation(Activation.SIGMOID).nOut(3).build(), "rnn")
                    .setInputTypes(InputType.recurrent(10))
                    .setOutputs("out")
                    .build();

            ComputationGraph cg = new ComputationGraph(conf);
            cg.init();


            INDArray[] input = new INDArray[]{Nd4j.zeros(1, 10, 5)};

            for (boolean train : new boolean[]{false, true}) {
                cg.clear();
                cg.feedForward(input, train);
            }

            cg.setInputs(input);
            cg.setLabels(Nd4j.rand(1, 3, 5));
            cg.computeGradientAndScore();
        }
    }

    @Test
    public void testWithPreprocessorsMLN() {
        for (WorkspaceMode wm : WorkspaceMode.values()) {
            System.out.println(wm);
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .trainingWorkspaceMode(wm)
                    .inferenceWorkspaceMode(wm)
                    .list()
                    .layer(new GravesLSTM.Builder().nIn(10).nOut(5).build())
                    .layer(new GravesLSTM.Builder().nIn(5).nOut(8).build())
                    .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nOut(3).build())
                    .inputPreProcessor(0, new DupPreProcessor())
                    .setInputType(InputType.recurrent(10))
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();


            INDArray input = Nd4j.zeros(1, 10, 5);

            for (boolean train : new boolean[]{false, true}) {
                net.clear();
                net.feedForward(input, train);
            }

            net.setInput(input);
            net.setLabels(Nd4j.rand(1, 3, 5));
            net.computeGradientAndScore();
        }
    }

    public static class DupPreProcessor implements InputPreProcessor {
        @Override
        public INDArray preProcess(INDArray input, int miniBatchSize, LayerWorkspaceMgr mgr) {
            return mgr.dup(ArrayType.ACTIVATIONS, input);
        }

        @Override
        public INDArray backprop(INDArray output, int miniBatchSize, LayerWorkspaceMgr workspaceMgr) {
            return workspaceMgr.dup(ArrayType.ACTIVATION_GRAD, output);
        }

        @Override
        public InputPreProcessor clone() {
            return new DupPreProcessor();
        }

        @Override
        public InputType getOutputType(InputType inputType) {
            return inputType;
        }

        @Override
        public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
            return new Pair<>(maskArray, currentMaskState);
        }
    }


    @Test
    public void testRnnTimeStep() {
        for (WorkspaceMode ws : WorkspaceMode.values()) {
            for (int i = 0; i < 3; i++) {

                System.out.println("Starting test: " + ws + " - " + i);

                NeuralNetConfiguration.ListBuilder b = new NeuralNetConfiguration.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.TANH)
                        .inferenceWorkspaceMode(ws)
                        .trainingWorkspaceMode(ws)
                        .list();

                ComputationGraphConfiguration.GraphBuilder gb = new NeuralNetConfiguration.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.TANH)
                        .inferenceWorkspaceMode(ws)
                        .trainingWorkspaceMode(ws)
                        .graphBuilder()
                        .addInputs("in");

                switch (i) {
                    case 0:
                        b.layer(new SimpleRnn.Builder().nIn(10).nOut(10).build());
                        b.layer(new SimpleRnn.Builder().nIn(10).nOut(10).build());

                        gb.addLayer("0", new SimpleRnn.Builder().nIn(10).nOut(10).build(), "in");
                        gb.addLayer("1", new SimpleRnn.Builder().nIn(10).nOut(10).build(), "0");
                        break;
                    case 1:
                        b.layer(new LSTM.Builder().nIn(10).nOut(10).build());
                        b.layer(new LSTM.Builder().nIn(10).nOut(10).build());

                        gb.addLayer("0", new LSTM.Builder().nIn(10).nOut(10).build(), "in");
                        gb.addLayer("1", new LSTM.Builder().nIn(10).nOut(10).build(), "0");
                        break;
                    case 2:
                        b.layer(new GravesLSTM.Builder().nIn(10).nOut(10).build());
                        b.layer(new GravesLSTM.Builder().nIn(10).nOut(10).build());

                        gb.addLayer("0", new GravesLSTM.Builder().nIn(10).nOut(10).build(), "in");
                        gb.addLayer("1", new GravesLSTM.Builder().nIn(10).nOut(10).build(), "0");
                        break;
                    default:
                        throw new RuntimeException();
                }

                b.layer(new RnnOutputLayer.Builder().nIn(10).nOut(10).build());
                gb.addLayer("out", new RnnOutputLayer.Builder().nIn(10).nOut(10).build(), "1");
                gb.setOutputs("out");

                MultiLayerConfiguration conf = b.build();
                ComputationGraphConfiguration conf2 = gb.build();


                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                ComputationGraph net2 = new ComputationGraph(conf2);
                net2.init();

                for (int j = 0; j < 3; j++) {
                    net.rnnTimeStep(Nd4j.rand(new int[]{3, 10, 5}));
                }

                for (int j = 0; j < 3; j++) {
                    net2.rnnTimeStep(Nd4j.rand(new int[]{3, 10, 5}));
                }
            }
        }
    }

    @Test
    public void testTbpttFit() {
        for (WorkspaceMode ws : WorkspaceMode.values()) {
            for (int i = 0; i < 3; i++) {

                System.out.println("Starting test: " + ws + " - " + i);

                NeuralNetConfiguration.ListBuilder b = new NeuralNetConfiguration.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.TANH)
                        .inferenceWorkspaceMode(ws)
                        .trainingWorkspaceMode(ws)
                        .list();

                ComputationGraphConfiguration.GraphBuilder gb = new NeuralNetConfiguration.Builder()
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.TANH)
                        .inferenceWorkspaceMode(ws)
                        .trainingWorkspaceMode(ws)
                        .graphBuilder()
                        .addInputs("in");

                switch (i) {
                    case 0:
                        b.layer(new SimpleRnn.Builder().nIn(10).nOut(10).build());
                        b.layer(new SimpleRnn.Builder().nIn(10).nOut(10).build());

                        gb.addLayer("0", new SimpleRnn.Builder().nIn(10).nOut(10).build(), "in");
                        gb.addLayer("1", new SimpleRnn.Builder().nIn(10).nOut(10).build(), "0");
                        break;
                    case 1:
                        b.layer(new LSTM.Builder().nIn(10).nOut(10).build());
                        b.layer(new LSTM.Builder().nIn(10).nOut(10).build());

                        gb.addLayer("0", new LSTM.Builder().nIn(10).nOut(10).build(), "in");
                        gb.addLayer("1", new LSTM.Builder().nIn(10).nOut(10).build(), "0");
                        break;
                    case 2:
                        b.layer(new GravesLSTM.Builder().nIn(10).nOut(10).build());
                        b.layer(new GravesLSTM.Builder().nIn(10).nOut(10).build());

                        gb.addLayer("0", new GravesLSTM.Builder().nIn(10).nOut(10).build(), "in");
                        gb.addLayer("1", new GravesLSTM.Builder().nIn(10).nOut(10).build(), "0");
                        break;
                    default:
                        throw new RuntimeException();
                }

                b.layer(new RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(10).nOut(10).build());
                gb.addLayer("out", new RnnOutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE)
                        .nIn(10).nOut(10).build(), "1");
                gb.setOutputs("out");

                MultiLayerConfiguration conf = b
                        .backpropType(BackpropType.TruncatedBPTT)
                        .tBPTTLength(5)
                        .build();

                ComputationGraphConfiguration conf2 = gb
                        .backpropType(BackpropType.TruncatedBPTT)
                        .tBPTTForwardLength(5).tBPTTBackwardLength(5)
                        .build();


                MultiLayerNetwork net = new MultiLayerNetwork(conf);
                net.init();

                ComputationGraph net2 = new ComputationGraph(conf2);
                net2.init();

                for (int j = 0; j < 3; j++) {
                    net.fit(Nd4j.rand(new int[]{3, 10, 20}), Nd4j.rand(new int[]{3, 10, 20}));
                }

                for (int j = 0; j < 3; j++) {
                    net2.fit(new DataSet(Nd4j.rand(new int[]{3, 10, 20}), Nd4j.rand(new int[]{3, 10, 20})));
                }
            }
        }
    }

    @Test
    public void testScalarOutputCase() {
        for (WorkspaceMode ws : WorkspaceMode.values()) {
            log.info("WorkspaceMode = " + ws);

            Nd4j.getRandom().setSeed(12345);
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .weightInit(WeightInit.XAVIER)
                    .seed(12345)
                    .trainingWorkspaceMode(ws).inferenceWorkspaceMode(ws)
                    .list()
                    .layer(new OutputLayer.Builder().nIn(3).nOut(1).activation(Activation.SIGMOID).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            INDArray input = Nd4j.linspace(1, 3, 3);
            INDArray out = net.output(input);
            INDArray out2 = net.output(input);

            assertEquals(out2, out);

            assertFalse(out.isAttached());
            assertFalse(out2.isAttached());

            Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
        }
    }


    @Test
    public void testWorkspaceSetting() {

        for (WorkspaceMode wsm : WorkspaceMode.values()) {
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .weightInit(WeightInit.XAVIER)
                    .seed(12345)
                    .list()
                    .layer(new OutputLayer.Builder().nIn(3).nOut(1).activation(Activation.SIGMOID).build())
                    .trainingWorkspaceMode(wsm).inferenceWorkspaceMode(wsm)
                    .build();

            assertEquals(wsm, conf.getTrainingWorkspaceMode());
            assertEquals(wsm, conf.getInferenceWorkspaceMode());


            MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                    .weightInit(WeightInit.XAVIER)
                    .seed(12345)
                    .trainingWorkspaceMode(wsm).inferenceWorkspaceMode(wsm)
                    .list()
                    .layer(new OutputLayer.Builder().nIn(3).nOut(1).activation(Activation.SIGMOID).build())
                    .build();

            assertEquals(wsm, conf2.getTrainingWorkspaceMode());
            assertEquals(wsm, conf2.getInferenceWorkspaceMode());
        }
    }


    @Test
    public void testClearing() {
        for(WorkspaceMode wsm : WorkspaceMode.values()) {
            ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                    .updater(new Adam())
                    .inferenceWorkspaceMode(wsm)
                    .trainingWorkspaceMode(wsm)
                    .graphBuilder()
                    .addInputs("in")
                    .setInputTypes(InputType.recurrent(200))
                    .addLayer("embeddings", new EmbeddingLayer.Builder().nIn(200).nOut(50).build(), "in")
                    .addLayer("a", new GravesLSTM.Builder().nOut(300).activation(Activation.HARDTANH).build(), "embeddings")
                    .addVertex("b", new LastTimeStepVertex("in"), "a")
                    .addLayer("c", new DenseLayer.Builder().nOut(300).activation(Activation.HARDTANH).build(), "b")
                    .addLayer("output", new LossLayer.Builder().lossFunction(LossFunctions.LossFunction.COSINE_PROXIMITY).build(), "c")
                    .setOutputs("output")
                    .build();


            final ComputationGraph computationGraph = new ComputationGraph(config);
            computationGraph.init();
            computationGraph.setListeners(new ScoreIterationListener(1));

            WSTestDataSetIterator iterator = new WSTestDataSetIterator();
            computationGraph.fit(iterator);
        }
    }
}
