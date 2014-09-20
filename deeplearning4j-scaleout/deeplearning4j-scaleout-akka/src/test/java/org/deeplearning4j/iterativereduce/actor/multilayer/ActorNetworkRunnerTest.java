package org.deeplearning4j.iterativereduce.actor.multilayer;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.test.TestDataSetIterator;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.scaleout.conf.Conf;
import org.junit.Test;
import org.nd4j.linalg.transformation.MatrixTransform;
import org.nd4j.linalg.transformation.MultiplyScalar;

import java.io.IOException;
import java.util.Collections;

import static org.junit.Assert.assertEquals;

public class ActorNetworkRunnerTest {

    @Test
    public void testNumTimesDataSetHit() throws IOException {
        MnistDataSetIterator mnist = new MnistDataSetIterator(20, 100);
        TestDataSetIterator iter = new TestDataSetIterator(mnist);
        ActorNetworkRunner runner = new ActorNetworkRunner(iter);

        System.setProperty("akka.remote.netty.tcp.hostname","localhost");

        NeuralNetConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .nIn(784).nOut(10).build();

        Conf conf = new Conf();
        conf.setConf(conf2);
        conf.getConf().setFinetuneEpochs(1000);
        conf.getConf().setPretrainLearningRate(0.0001f);
        conf.setLayerSizes(new int[]{500,250,100});
        conf.setMultiLayerClazz(DBN.class);
        conf.getConf().setnOut(10);
        conf.getConf().setFinetuneLearningRate(0.0001f);
        conf.getConf().setnIn(784);
        conf.getConf().setL2(0.001f);
        conf.getConf().setMomentum(0);
        conf.setSplit(10);
        //conf.setRenderWeightEpochs(100);
        conf.getConf().setUseRegularization(false);
        conf.setDeepLearningParams(new Object[]{1,0.0001,1000});
        runner.setup(conf);

        runner.train();




    }

    @Test
    public void testClusterSize() throws Exception {
        MnistDataSetIterator mnist = new MnistDataSetIterator(5, 5);
        TestDataSetIterator iter = new TestDataSetIterator(mnist);
        ActorNetworkRunner runner = new ActorNetworkRunner(iter);

        NeuralNetConfiguration conf2 = new NeuralNetConfiguration.Builder()
                .nIn(784).nOut(10).build();



        Conf conf = new Conf();
        conf.getConf().setFinetuneEpochs(1);
        conf.getConf().setPretrainEpochs(1);
        conf.getConf().setPretrainLearningRate(0.0001f);
        conf.setLayerSizes(new int[]{500,250,100});
        conf.setMultiLayerClazz(DBN.class);
        conf.getConf().setnOut(10);
        conf.getConf().setFinetuneLearningRate(0.0001f);
        conf.getConf().setnIn(784);
        conf.getConf().setL2(0.001f);
        conf.getConf().setMomentum(0);
        conf.setWeightTransforms(Collections.singletonMap(0,(MatrixTransform) new MultiplyScalar(1000)));
        conf.setSplit(10);
        //conf.setRenderWeightEpochs(100);
        conf.getConf().setUseRegularization(false);
        conf.setDeepLearningParams(new Object[]{1,0.0001,1});
        runner.setStateTrackerPort(1100);
        runner.setup(conf);



        ActorNetworkRunner worker = new ActorNetworkRunner(runner.getMasterAddress().toString(),"worker");
        worker.setStateTrackerPort(1100);
        worker.setup(conf);


        assertEquals(true,runner.getStateTracker().numWorkers() == worker.getStateTracker().numWorkers());


        runner.train();
        worker.shutdown();
        runner.shutdown();



    }


}
