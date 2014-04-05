package org.deeplearning4j.iterativereduce.actor.multilayer;

import static org.junit.Assert.*;

import java.io.IOException;
import java.util.Collections;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.test.TestDataSetIterator;
import org.deeplearning4j.transformation.MatrixTransform;
import org.deeplearning4j.transformation.MultiplyScalar;
import org.junit.Ignore;
import org.junit.Test;

public class ActorNetworkRunnerTest {

    @Test
    @Ignore
    public void testNumTimesDataSetHit() throws IOException {
        MnistDataSetIterator mnist = new MnistDataSetIterator(20, 100);
        TestDataSetIterator iter = new TestDataSetIterator(mnist);
        ActorNetworkRunner runner = new ActorNetworkRunner(iter);
        Conf conf = new Conf();
        conf.setFinetuneEpochs(1000);
        conf.setPretrainLearningRate(0.0001);
        conf.setLayerSizes(new int[]{500,250,100});
        conf.setMultiLayerClazz(DBN.class);
        conf.setnOut(10);
        conf.setFinetuneLearningRate(0.0001);
        conf.setnIn(784);
        conf.setL2(0.001);
        conf.setMomentum(0);
        conf.setWeightTransforms(Collections.singletonMap(0,(MatrixTransform) new MultiplyScalar(1000)));
        conf.setSplit(10);
        //conf.setRenderWeightEpochs(100);
        conf.setUseRegularization(false);
        conf.setDeepLearningParams(new Object[]{1,0.0001,1000});
        runner.setup(conf);

        runner.train();


    }

    @Test
    public void testClusterSize() throws Exception {
        MnistDataSetIterator mnist = new MnistDataSetIterator(5, 5);
        TestDataSetIterator iter = new TestDataSetIterator(mnist);
        ActorNetworkRunner runner = new ActorNetworkRunner(iter);
        Conf conf = new Conf();
        conf.setFinetuneEpochs(1);
        conf.setPretrainEpochs(1);
        conf.setPretrainLearningRate(0.0001);
        conf.setLayerSizes(new int[]{500,250,100});
        conf.setMultiLayerClazz(DBN.class);
        conf.setnOut(10);
        conf.setFinetuneLearningRate(0.0001);
        conf.setnIn(784);
        conf.setL2(0.001);
        conf.setMomentum(0);
        conf.setWeightTransforms(Collections.singletonMap(0,(MatrixTransform) new MultiplyScalar(1000)));
        conf.setSplit(10);
        //conf.setRenderWeightEpochs(100);
        conf.setUseRegularization(false);
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
