package org.deeplearning4j.example.deepautoencoder;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.iterativereduce.actor.core.DefaultModelSaver;
import org.deeplearning4j.iterativereduce.actor.deepautoencoder.DeepAutoEncoderDistributedTrainer;
import org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunner;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.deepautoencoder.DeepAutoEncoderHazelCastStateTracker;
import org.deeplearning4j.nn.OutputLayer;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collections;

/**
 * Created by agibsonccc on 5/25/14.
 */
public class DistributedDeepAutoEncoderExample {

    private static Logger log = LoggerFactory.getLogger(DistributedDeepAutoEncoderExample.class);

    public static void main(String[] args) throws Exception {

        DBN d = SerializationUtils.readObject(new File(args[0]));
        //batches of 10, 60000 examples total
        DataSetIterator iter = new MnistDataSetIterator(80,60000,false);

        Conf c = new Conf();
        c.setFinetuneEpochs(10000);
        c.setFinetuneLearningRate(1e-2);
        c.setPretrainLearningRate(1e-1);
        c.setNormalizeCodeLayer(false);
        c.setRoundCodeLayer(true);
        c.setSplit(100);
        c.setSparsity(1e-1);
        c.setMomentum(9e-1);
        c.setMultiLayerClazz(DBN.class);
        c.setUseRegularization(true);
        c.setL2(2e-2);

        c.setOutputLayerLossFunction(OutputLayer.LossFunction.RMSE_XENT);
        c.setOutputActivationFunction(Activations.sigmoid());
         DeepAutoEncoderHazelCastStateTracker tracker = new DeepAutoEncoderHazelCastStateTracker();
        tracker.moveToFinetune();


        DeepAutoEncoderDistributedTrainer runner = new DeepAutoEncoderDistributedTrainer("master",iter,d);
        runner.setStateTracker(tracker);
        runner.setModelSaver(new DefaultModelSaver(new File("mnist-example-deepautoencoder.ser")));
        runner.setup(c);
        runner.train();

    }


}
