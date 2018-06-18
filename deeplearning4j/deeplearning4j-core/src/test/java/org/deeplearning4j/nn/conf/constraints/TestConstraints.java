package org.deeplearning4j.nn.conf.constraints;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.TestUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.constraint.MaxNormConstraint;
import org.deeplearning4j.nn.conf.constraint.MinMaxNormConstraint;
import org.deeplearning4j.nn.conf.constraint.NonNegativeConstraint;
import org.deeplearning4j.nn.conf.constraint.UnitNormConstraint;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class TestConstraints extends BaseDL4JTest {

    @Test
    public void testLayerRecurrentConstraints() throws Exception {

        LayerConstraint[] constraints = new LayerConstraint[]{
                new MaxNormConstraint(0.5, 1),
                new MinMaxNormConstraint(0.3, 0.4, 1.0, 1),
                new NonNegativeConstraint(),
                new UnitNormConstraint(1)
        };

        for (LayerConstraint lc : constraints) {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .updater(new Sgd(0.0))
                    .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 5))
                    .list()
                    .layer(new LSTM.Builder().nIn(12).nOut(10)
                            .constrainRecurrent(lc).build())
                    .layer(new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(10).nOut(8).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            LayerConstraint exp = lc.clone();
            assertEquals(exp.toString(), net.getLayer(0).conf().getLayer().getConstraints().get(0).toString());

            INDArray input = Nd4j.rand(3, 12);
            INDArray labels = Nd4j.rand(3, 8);

            net.fit(input, labels);

            INDArray RW0 = net.getParam("0_RW");


            if (lc instanceof MaxNormConstraint) {
                assertTrue(RW0.norm2(1).maxNumber().doubleValue() <= 0.5);

            } else if (lc instanceof MinMaxNormConstraint) {
                assertTrue(RW0.norm2(1).minNumber().doubleValue() >= 0.3);
                assertTrue(RW0.norm2(1).maxNumber().doubleValue() <= 0.4);
            } else if (lc instanceof NonNegativeConstraint) {
                assertTrue(RW0.minNumber().doubleValue() >= 0.0);
            } else if (lc instanceof UnitNormConstraint) {
                assertEquals(RW0.norm2(1).minNumber().doubleValue(), 1.0, 1e-6);
                assertEquals(RW0.norm2(1).maxNumber().doubleValue(), 1.0, 1e-6);
            }

            TestUtils.testModelSerialization(net);
        }
    }

    @Test
    public void testLayerBiasConstraints() throws Exception {

        LayerConstraint[] constraints = new LayerConstraint[]{
                new MaxNormConstraint(0.5, 1),
                new MinMaxNormConstraint(0.3, 0.4, 1.0, 1),
                new NonNegativeConstraint(),
                new UnitNormConstraint(1)
        };

        for (LayerConstraint lc : constraints) {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .updater(new Sgd(0.0))
                    .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 5))
                    .biasInit(10.0)
                    .list()
                    .layer(new DenseLayer.Builder().nIn(12).nOut(10)
                            .constrainBias(lc).build())
                    .layer(new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(10).nOut(8).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            LayerConstraint exp = lc.clone();
            assertEquals(exp.toString(), net.getLayer(0).conf().getLayer().getConstraints().get(0).toString());

            INDArray input = Nd4j.rand(3, 12);
            INDArray labels = Nd4j.rand(3, 8);

            net.fit(input, labels);

            INDArray b0 = net.getParam("0_b");


            if (lc instanceof MaxNormConstraint) {
                assertTrue(b0.norm2(1).maxNumber().doubleValue() <= 0.5);

            } else if (lc instanceof MinMaxNormConstraint) {
                assertTrue(b0.norm2(1).minNumber().doubleValue() >= 0.3);
                assertTrue(b0.norm2(1).maxNumber().doubleValue() <= 0.4);
            } else if (lc instanceof NonNegativeConstraint) {
                assertTrue(b0.minNumber().doubleValue() >= 0.0);
            } else if (lc instanceof UnitNormConstraint) {
                assertEquals(b0.norm2(1).minNumber().doubleValue(), 1.0, 1e-6);
                assertEquals(b0.norm2(1).maxNumber().doubleValue(), 1.0, 1e-6);
            }

            TestUtils.testModelSerialization(net);
        }
    }

    @Test
    public void testLayerWeightsConstraints() throws Exception {

        LayerConstraint[] constraints = new LayerConstraint[]{
                new MaxNormConstraint(0.5, 1),
                new MinMaxNormConstraint(0.3, 0.4, 1.0, 1),
                new NonNegativeConstraint(),
                new UnitNormConstraint(1)
        };

        for (LayerConstraint lc : constraints) {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .updater(new Sgd(0.0))
                    .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 5))
                    .list()
                    .layer(new DenseLayer.Builder().nIn(12).nOut(10)
                            .constrainWeights(lc).build())
                    .layer(new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(10).nOut(8).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            LayerConstraint exp = lc.clone();
            assertEquals(exp.toString(), net.getLayer(0).conf().getLayer().getConstraints().get(0).toString());

            INDArray input = Nd4j.rand(3, 12);
            INDArray labels = Nd4j.rand(3, 8);

            net.fit(input, labels);

            INDArray w0 = net.getParam("0_W");


            if (lc instanceof MaxNormConstraint) {
                assertTrue(w0.norm2(1).maxNumber().doubleValue() <= 0.5);

            } else if (lc instanceof MinMaxNormConstraint) {
                assertTrue(w0.norm2(1).minNumber().doubleValue() >= 0.3);
                assertTrue(w0.norm2(1).maxNumber().doubleValue() <= 0.4);
            } else if (lc instanceof NonNegativeConstraint) {
                assertTrue(w0.minNumber().doubleValue() >= 0.0);
            } else if (lc instanceof UnitNormConstraint) {
                assertEquals(w0.norm2(1).minNumber().doubleValue(), 1.0, 1e-6);
                assertEquals(w0.norm2(1).maxNumber().doubleValue(), 1.0, 1e-6);
            }

            TestUtils.testModelSerialization(net);
        }
    }

    @Test
    public void testLayerWeightsAndBiasConstraints() throws Exception {

        LayerConstraint[] constraints = new LayerConstraint[]{
                new MaxNormConstraint(0.5, 1),
                new MinMaxNormConstraint(0.3, 0.4, 1.0, 1),
                new NonNegativeConstraint(),
                new UnitNormConstraint(1)
        };

        for (LayerConstraint lc : constraints) {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .updater(new Sgd(0.0))
                    .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 5))
                    .biasInit(0.2)
                    .list()
                    .layer(new DenseLayer.Builder().nIn(12).nOut(10)
                            .constrainAllParameters(lc).build())
                    .layer(new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(10).nOut(8).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            LayerConstraint exp = lc.clone();
            assertEquals(exp.toString(), net.getLayer(0).conf().getLayer().getConstraints().get(0).toString());

            INDArray input = Nd4j.rand(3, 12);
            INDArray labels = Nd4j.rand(3, 8);

            net.fit(input, labels);

            INDArray w0 = net.getParam("0_W");
            INDArray b0 = net.getParam("0_b");


            if (lc instanceof MaxNormConstraint) {
                assertTrue(w0.norm2(1).maxNumber().doubleValue() <= 0.5);
                assertTrue(b0.norm2(1).maxNumber().doubleValue() <= 0.5);

            } else if (lc instanceof MinMaxNormConstraint) {
                assertTrue(w0.norm2(1).minNumber().doubleValue() >= 0.3);
                assertTrue(w0.norm2(1).maxNumber().doubleValue() <= 0.4);
                assertTrue(b0.norm2(1).minNumber().doubleValue() >= 0.3);
                assertTrue(b0.norm2(1).maxNumber().doubleValue() <= 0.4);
            } else if (lc instanceof NonNegativeConstraint) {
                assertTrue(w0.minNumber().doubleValue() >= 0.0);
                assertTrue(b0.minNumber().doubleValue() >= 0.0);
            } else if (lc instanceof UnitNormConstraint) {
                assertEquals(w0.norm2(1).minNumber().doubleValue(), 1.0, 1e-6);
                assertEquals(w0.norm2(1).maxNumber().doubleValue(), 1.0, 1e-6);
                assertEquals(b0.norm2(1).minNumber().doubleValue(), 1.0, 1e-6);
                assertEquals(b0.norm2(1).maxNumber().doubleValue(), 1.0, 1e-6);
            }

            TestUtils.testModelSerialization(net);
        }
    }


    @Test
    public void testLayerWeightsAndBiasSeparateConstraints() throws Exception {

        LayerConstraint[] constraints = new LayerConstraint[]{
                new MaxNormConstraint(0.5, 1),
                new MinMaxNormConstraint(0.3, 0.4, 1.0, 1),
                new NonNegativeConstraint(),
                new UnitNormConstraint(1)
        };

        for (LayerConstraint lc : constraints) {

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .updater(new Sgd(0.0))
                    .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0, 5))
                    .biasInit(0.2)
                    .list()
                    .layer(new DenseLayer.Builder().nIn(12).nOut(10)
                            .constrainWeights(lc).constrainBias(lc).build())
                    .layer(new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(10).nOut(8).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            LayerConstraint exp = lc.clone();
            assertEquals(exp.toString(), net.getLayer(0).conf().getLayer().getConstraints().get(0).toString());

            INDArray input = Nd4j.rand(3, 12);
            INDArray labels = Nd4j.rand(3, 8);

            net.fit(input, labels);

            INDArray w0 = net.getParam("0_W");
            INDArray b0 = net.getParam("0_b");


            if (lc instanceof MaxNormConstraint) {
                assertTrue(w0.norm2(1).maxNumber().doubleValue() <= 0.5);
                assertTrue(b0.norm2(1).maxNumber().doubleValue() <= 0.5);

            } else if (lc instanceof MinMaxNormConstraint) {
                assertTrue(w0.norm2(1).minNumber().doubleValue() >= 0.3);
                assertTrue(w0.norm2(1).maxNumber().doubleValue() <= 0.4);
                assertTrue(b0.norm2(1).minNumber().doubleValue() >= 0.3);
                assertTrue(b0.norm2(1).maxNumber().doubleValue() <= 0.4);
            } else if (lc instanceof NonNegativeConstraint) {
                assertTrue(w0.minNumber().doubleValue() >= 0.0);
                assertTrue(b0.minNumber().doubleValue() >= 0.0);
            } else if (lc instanceof UnitNormConstraint) {
                assertEquals(w0.norm2(1).minNumber().doubleValue(), 1.0, 1e-6);
                assertEquals(w0.norm2(1).maxNumber().doubleValue(), 1.0, 1e-6);
                assertEquals(b0.norm2(1).minNumber().doubleValue(), 1.0, 1e-6);
                assertEquals(b0.norm2(1).maxNumber().doubleValue(), 1.0, 1e-6);
            }

            TestUtils.testModelSerialization(net);
        }
    }

        @Test
    public void testModelConstraints() throws Exception {

        LayerConstraint[] constraints = new LayerConstraint[]{
                new MaxNormConstraint(0.5, 1),
                new MinMaxNormConstraint(0.3, 0.4, 1.0, 1),
                new NonNegativeConstraint(),
                new UnitNormConstraint(1)
        };

        for(LayerConstraint lc : constraints){

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .constrainWeights(lc)
                    .updater(new Sgd(0.0))
                    .weightInit(WeightInit.DISTRIBUTION).dist(new NormalDistribution(0,5))
                    .biasInit(1)
                    .list()
                    .layer(new DenseLayer.Builder().nIn(12).nOut(10).build())
                    .layer(new OutputLayer.Builder().lossFunction(LossFunctions.LossFunction.MSE).nIn(10).nOut(8).build())
                    .build();

            MultiLayerNetwork net = new MultiLayerNetwork(conf);
            net.init();

            LayerConstraint exp = lc.clone();
            assertEquals(exp.toString(), net.getLayer(0).conf().getLayer().getConstraints().get(0).toString());
            assertEquals(exp.toString(), net.getLayer(1).conf().getLayer().getConstraints().get(0).toString());

            INDArray input = Nd4j.rand(3, 12);
            INDArray labels = Nd4j.rand(3, 8);

            net.fit(input, labels);

            INDArray w0 = net.getParam("0_W");
            INDArray w1 = net.getParam("1_W");

            if(lc instanceof MaxNormConstraint){
                assertTrue(w0.norm2(1).maxNumber().doubleValue() <= 0.5 );
                assertTrue(w1.norm2(1).maxNumber().doubleValue() <= 0.5 );
            } else if(lc instanceof MinMaxNormConstraint){
                assertTrue(w0.norm2(1).minNumber().doubleValue() >= 0.3 );
                assertTrue(w0.norm2(1).maxNumber().doubleValue() <= 0.4 );
                assertTrue(w1.norm2(1).minNumber().doubleValue() >= 0.3 );
                assertTrue(w1.norm2(1).maxNumber().doubleValue() <= 0.4 );
            } else if(lc instanceof NonNegativeConstraint ){
                assertTrue(w0.minNumber().doubleValue() >= 0.0 );
            } else if(lc instanceof UnitNormConstraint ){
                assertEquals(w0.norm2(1).minNumber().doubleValue(), 1.0, 1e-6 );
                assertEquals(w0.norm2(1).maxNumber().doubleValue(), 1.0, 1e-6 );
                assertEquals(w1.norm2(1).minNumber().doubleValue(), 1.0, 1e-6 );
                assertEquals(w1.norm2(1).maxNumber().doubleValue(), 1.0, 1e-6 );
            }

            TestUtils.testModelSerialization(net);
        }
    }


    @Test
    public void testConstraints(){

        double learningRate = 0.001;
        int nIn = 10;
        int lstmLayerSize = 32;

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.RELU_UNIFORM)
                .updater(new RmsProp(learningRate))
                .graphBuilder()
                .addInputs("input_lstm", "input_cpc")
                .addLayer("first_lstm_layer",
                        new LSTM.Builder()
                                .nIn(nIn)
                                .nOut(lstmLayerSize)
                                .activation(Activation.RELU)
                                .constrainWeights(new NonNegativeConstraint())
                                .build(),
                        "input_lstm")
                .addVertex("lastTimeStep", new LastTimeStepVertex("input_lstm"), "first_lstm_layer")
                .addVertex("merge", new MergeVertex(),
                        "lastTimeStep", "input_cpc")
                .addLayer("dense",
                        new DenseLayer.Builder()
                                .constrainWeights(new NonNegativeConstraint())
                                .nIn(lstmLayerSize  + 1)
                                .nOut(lstmLayerSize/2)
                                .activation(Activation.RELU)
                                .build(),
                        "merge")
                .addLayer("second_dense",
                        new DenseLayer.Builder()
                                .constrainWeights(new NonNegativeConstraint())
                                .nIn(lstmLayerSize/2)
                                .nOut(lstmLayerSize/8)
                                .activation(Activation.RELU)
                                .build(),
                        "dense")
                .addLayer("output_layer",
                        new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                                .constrainWeights(new NonNegativeConstraint())
                                .nIn(lstmLayerSize/8)
                                .nOut(1)
                                .activation(Activation.IDENTITY)
                                .build(),
                        "second_dense")
                .setOutputs("output_layer")
                .backpropType(BackpropType.Standard)
                .pretrain(false)
                .backprop(true)
                .build();

        ComputationGraph g = new ComputationGraph(conf);
        g.init();


        for( int i=0; i<100; i++ ){
            INDArray in1 = Nd4j.rand(new int[]{1, nIn, 5});
            INDArray in2 = Nd4j.rand(new int[]{1, 1});
            INDArray label = Nd4j.rand(new int[]{1, 1});
            g.fit(new INDArray[]{in1, in2}, new INDArray[]{label});

            for(Map.Entry<String,INDArray> e : g.paramTable().entrySet()){
                if(!e.getKey().contains("W")){
                    continue;
                }

                double min = e.getValue().minNumber().doubleValue();
                assertTrue( min >= 0.0);
            }
        }
    }


}
