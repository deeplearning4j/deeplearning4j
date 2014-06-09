package org.deeplearning4j.example.deepautoencoder;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.iterativereduce.actor.core.DefaultModelSaver;
import org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunner;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.util.SerializationUtils;

import java.io.File;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Equivalent multi threaded example
 * from the {@link org.deeplearning4j.example.mnist.MnistExample}
 *
 * @author Adam Gibson
 *
 */
public class MnistExampleMultiThreaded {

    /**
     * @param args
     */
    public static void main(String[] args) throws Exception {
        //batches of 10, 60000 examples total, don't binarize
        DataSetIterator iter = new MnistDataSetIterator(80,60000,false);
        int codeLayer = 3;
        Conf c = new Conf();
        c.setFinetuneEpochs(10000);
        c.setPretrainEpochs(100);
        c.setFinetuneLearningRate(1e-1);
        c.setPretrainLearningRate(1e-1);
        c.setLayerSizes(new int[]{1000, 500,250,30});
        c.setnIn(784);
        c.setUseRegularization(true);
        c.setL2(2e-4);
        c.setNormalizeZeroMeanAndUnitVariance(false);
        c.setScale(true);
        c.setMomentum(9e-1);
        c.setDropOut(0);
        Map<Integer,RBM.HiddenUnit> hiddenUnitMap = new HashMap<>();
        hiddenUnitMap.put(codeLayer, RBM.HiddenUnit.GAUSSIAN);
        c.setHiddenUnitByLayer(Collections.singletonMap(codeLayer, RBM.HiddenUnit.GAUSSIAN));
        c.setActivationFunctionForLayer(Collections.singletonMap(codeLayer,Activations.sigmoid()));
        c.setSplit(100);
        c.setLearningRateForLayer(Collections.singletonMap(codeLayer,1e-1));
        c.setSparsity(0);

        c.setnOut(10);
        c.setMultiLayerClazz(DBN.class);
        c.setDeepLearningParams(new Object[]{1,1e-1,100});
        ActorNetworkRunner runner = args.length < 1 ? new ActorNetworkRunner("master",iter) : new ActorNetworkRunner("master",iter, (DBN) SerializationUtils.readObject(new File(args[0])));
        runner.setModelSaver(new DefaultModelSaver(new File("mnist-example-deepautoencoder.ser")));
        runner.setup(c);
        runner.train();
    }

}
