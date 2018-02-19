package org.deeplearning4j.nn.misc;

import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.weightnoise.DropConnect;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ExponentialSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import static org.junit.Assert.assertEquals;

public class TestLrChanges extends BaseDL4JTest {

    @Test
    public void testChangeLrMLN(){
        //First: Set LR for a *single* layer and compare vs. equivalent net config
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .seed(12345)
                .list()
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).updater(new Adam(0.1)).build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).updater(new RmsProp(0.01)).build())
                .layer(new OutputLayer.Builder().nIn(10).nOut(10).updater(new NoOp()).lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        for( int i=0; i<10; i++ ){
            net.fit(Nd4j.rand(10,10), Nd4j.rand(10,10));
        }


        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .seed(12345)
                .list()
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).updater(new Adam(0.5)).build())    //0.5 LR
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).updater(new RmsProp(0.01)).build())
                .layer(new OutputLayer.Builder().nIn(10).nOut(10).updater(new NoOp()).lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();
        MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
        net2.init();
        net2.getUpdater().getStateViewArray().assign(net.getUpdater().getStateViewArray());
        conf2.setIterationCount(conf.getIterationCount());
        net2.setParams(net.params().dup());

        net.setLearningRate(0, 0.5);  //Set LR for layer 0 to 0.5

        assertEquals(conf, conf2);
        assertEquals(conf.toJson(), conf2.toJson());

        assertEquals(net.getUpdater().getStateViewArray(), net2.getUpdater().getStateViewArray());

        //Perform some parameter updates - check things are actually in sync...
        for( int i=0; i<3; i++ ){
            INDArray in = Nd4j.rand(10, 10);
            INDArray l = Nd4j.rand(10, 10);

            net.fit(in, l);
            net2.fit(in, l);
        }

        assertEquals(net.params(), net2.params());
        assertEquals(net.getUpdater().getStateViewArray(), net2.getUpdater().getStateViewArray());

        INDArray in1 = Nd4j.rand(10, 10);
        INDArray l1 = Nd4j.rand(10, 10);

        net.setInput(in1);
        net.setLabels(l1);
        net.computeGradientAndScore();

        net2.setInput(in1);
        net2.setLabels(l1);
        net2.computeGradientAndScore();

        assertEquals(net.score(), net2.score(), 1e-8);


        //Now: Set *all* LRs to say 0.3...
        MultiLayerConfiguration conf3 = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .seed(12345)
                .list()
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).updater(new Adam(0.3)).build())    //0.5 LR
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).updater(new RmsProp(0.3)).build())
                .layer(new OutputLayer.Builder().nIn(10).nOut(10).updater(new NoOp()).lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();
        MultiLayerNetwork net3 = new MultiLayerNetwork(conf3);
        net3.init();
        net3.getUpdater().getStateViewArray().assign(net.getUpdater().getStateViewArray());
        conf3.setIterationCount(conf.getIterationCount());
        net3.setParams(net.params().dup());

        net.setLearningRate(0.3);

        //Perform some parameter updates - check things are actually in sync...
        for( int i=0; i<3; i++ ){
            INDArray in = Nd4j.rand(10, 10);
            INDArray l = Nd4j.rand(10, 10);

            net.fit(in, l);
            net3.fit(in, l);
        }

        assertEquals(net.params(), net3.params());
        assertEquals(net.getUpdater().getStateViewArray(), net3.getUpdater().getStateViewArray());
    }

    @Test
    public void testChangeLrMLNSchedule(){
        //First: Set LR for a *single* layer and compare vs. equivalent net config
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .seed(12345)
                .updater(new Adam(0.1))
                .list()
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(new OutputLayer.Builder().nIn(10).nOut(10).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        for( int i=0; i<10; i++ ){
            net.fit(Nd4j.rand(10,10), Nd4j.rand(10,10));
        }


        MultiLayerConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .seed(12345)
                .updater(new Adam(new ExponentialSchedule(ScheduleType.ITERATION, 0.5, 0.8 )))
                .list()
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).build())
                .layer(new OutputLayer.Builder().nIn(10).nOut(10).build())
                .build();
        MultiLayerNetwork net2 = new MultiLayerNetwork(conf2);
        net2.init();
        net2.getUpdater().getStateViewArray().assign(net.getUpdater().getStateViewArray());
        conf2.setIterationCount(conf.getIterationCount());
        net2.setParams(net.params().dup());

        net.setLearningRate(new ExponentialSchedule(ScheduleType.ITERATION, 0.5, 0.8 ));  //Set LR for layer 0 to 0.5

        assertEquals(conf, conf2);
        assertEquals(conf.toJson(), conf2.toJson());

        assertEquals(net.getUpdater().getStateViewArray(), net2.getUpdater().getStateViewArray());

        //Perform some parameter updates - check things are actually in sync...
        for( int i=0; i<3; i++ ){
            INDArray in = Nd4j.rand(10, 10);
            INDArray l = Nd4j.rand(10, 10);

            net.fit(in, l);
            net2.fit(in, l);
        }

        assertEquals(net.params(), net2.params());
        assertEquals(net.getUpdater().getStateViewArray(), net2.getUpdater().getStateViewArray());
    }







    @Test
    public void testChangeLrCompGraph(){
        //First: Set LR for a *single* layer and compare vs. equivalent net config
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .seed(12345)
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new DenseLayer.Builder().nIn(10).nOut(10).updater(new Adam(0.1)).build(), "in")
                .addLayer("1", new DenseLayer.Builder().nIn(10).nOut(10).updater(new RmsProp(0.01)).build(), "0")
                .addLayer("2", new OutputLayer.Builder().nIn(10).nOut(10).updater(new NoOp()).lossFunction(LossFunctions.LossFunction.MSE).build(), "1")
                .setOutputs("2")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        for( int i=0; i<10; i++ ){
            net.fit(new DataSet(Nd4j.rand(10,10), Nd4j.rand(10,10)));
        }


        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .seed(12345)
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new DenseLayer.Builder().nIn(10).nOut(10).updater(new Adam(0.5)).build(), "in")  //0.5 LR
                .addLayer("1", new DenseLayer.Builder().nIn(10).nOut(10).updater(new RmsProp(0.01)).build(), "0")
                .addLayer("2", new OutputLayer.Builder().nIn(10).nOut(10).updater(new NoOp()).lossFunction(LossFunctions.LossFunction.MSE).build(), "1")
                .setOutputs("2")
                .build();
        ComputationGraph net2 = new ComputationGraph(conf2);
        net2.init();
        net2.getUpdater().getStateViewArray().assign(net.getUpdater().getStateViewArray());
        conf2.setIterationCount(conf.getIterationCount());
        net2.setParams(net.params().dup());

        net.setLearningRate("0", 0.5);  //Set LR for layer 0 to 0.5

        assertEquals(conf, conf2);
        assertEquals(conf.toJson(), conf2.toJson());

        assertEquals(net.getUpdater().getStateViewArray(), net2.getUpdater().getStateViewArray());

        //Perform some parameter updates - check things are actually in sync...
        for( int i=0; i<3; i++ ){
            INDArray in = Nd4j.rand(10, 10);
            INDArray l = Nd4j.rand(10, 10);

            net.fit(new DataSet(in, l));
            net2.fit(new DataSet(in, l));
        }

        assertEquals(net.params(), net2.params());
        assertEquals(net.getUpdater().getStateViewArray(), net2.getUpdater().getStateViewArray());

        INDArray in1 = Nd4j.rand(10, 10);
        INDArray l1 = Nd4j.rand(10, 10);

        net.setInputs(in1);
        net.setLabels(l1);
        net.computeGradientAndScore();

        net2.setInputs(in1);
        net2.setLabels(l1);
        net2.computeGradientAndScore();

        assertEquals(net.score(), net2.score(), 1e-8);


        //Now: Set *all* LRs to say 0.3...
        MultiLayerConfiguration conf3 = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .seed(12345)
                .list()
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).updater(new Adam(0.3)).build())    //0.5 LR
                .layer(new DenseLayer.Builder().nIn(10).nOut(10).updater(new RmsProp(0.3)).build())
                .layer(new OutputLayer.Builder().nIn(10).nOut(10).updater(new NoOp()).lossFunction(LossFunctions.LossFunction.MSE).build())
                .build();
        MultiLayerNetwork net3 = new MultiLayerNetwork(conf3);
        net3.init();
        net3.getUpdater().getStateViewArray().assign(net.getUpdater().getStateViewArray());
        conf3.setIterationCount(conf.getIterationCount());
        net3.setParams(net.params().dup());

        net.setLearningRate(0.3);

        //Perform some parameter updates - check things are actually in sync...
        for( int i=0; i<3; i++ ){
            INDArray in = Nd4j.rand(10, 10);
            INDArray l = Nd4j.rand(10, 10);

            net.fit(new DataSet(in, l));
            net3.fit(new DataSet(in, l));
        }

        assertEquals(net.params(), net3.params());
        assertEquals(net.getUpdater().getStateViewArray(), net3.getUpdater().getStateViewArray());
    }

    @Test
    public void testChangeLrCompGraphSchedule(){
        //First: Set LR for a *single* layer and compare vs. equivalent net config
        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .seed(12345)
                .updater(new Adam(0.1))
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in")
                .addLayer("1", new DenseLayer.Builder().nIn(10).nOut(10).build(), "0")
                .addLayer("2", new OutputLayer.Builder().nIn(10).nOut(10).build(), "1")
                .setOutputs("2")
                .build();

        ComputationGraph net = new ComputationGraph(conf);
        net.init();

        for( int i=0; i<10; i++ ){
            net.fit(new DataSet(Nd4j.rand(10,10), Nd4j.rand(10,10)));
        }


        ComputationGraphConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .activation(Activation.TANH)
                .seed(12345)
                .updater(new Adam(new ExponentialSchedule(ScheduleType.ITERATION, 0.5, 0.8 )))
                .graphBuilder()
                .addInputs("in")
                .addLayer("0", new DenseLayer.Builder().nIn(10).nOut(10).build(), "in")
                .addLayer("1", new DenseLayer.Builder().nIn(10).nOut(10).build(), "0")
                .layer("2", new OutputLayer.Builder().nIn(10).nOut(10).build(), "1")
                .setOutputs("2")
                .build();
        ComputationGraph net2 = new ComputationGraph(conf2);
        net2.init();
        net2.getUpdater().getStateViewArray().assign(net.getUpdater().getStateViewArray());
        conf2.setIterationCount(conf.getIterationCount());
        net2.setParams(net.params().dup());

        net.setLearningRate(new ExponentialSchedule(ScheduleType.ITERATION, 0.5, 0.8 ));  //Set LR for layer 0 to 0.5

        assertEquals(conf, conf2);
        assertEquals(conf.toJson(), conf2.toJson());

        assertEquals(net.getUpdater().getStateViewArray(), net2.getUpdater().getStateViewArray());

        //Perform some parameter updates - check things are actually in sync...
        for( int i=0; i<3; i++ ){
            INDArray in = Nd4j.rand(10, 10);
            INDArray l = Nd4j.rand(10, 10);

            net.fit(new DataSet(in, l));
            net2.fit(new DataSet(in, l));
        }

        assertEquals(net.params(), net2.params());
        assertEquals(net.getUpdater().getStateViewArray(), net2.getUpdater().getStateViewArray());
    }

}
