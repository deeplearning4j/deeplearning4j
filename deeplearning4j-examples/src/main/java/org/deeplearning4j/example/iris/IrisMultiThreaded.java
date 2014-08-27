package org.deeplearning4j.example.iris;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.SamplingDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.iterativereduce.actor.core.DefaultModelSaver;
import org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunner;
import org.deeplearning4j.linalg.api.activation.Activations;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.deeplearning4j.scaleout.conf.Conf;

import java.io.File;

/**
 * Adam Gibson
 */
public class IrisMultiThreaded {

    public static void main(String[] args) throws Exception {
        DataSetIterator iter = new IrisDataSetIterator(150, 150);
        DataSet next = iter.next();
       // next.normalizeZeroMeanZeroUnitVariance();
       next.normalizeZeroMeanZeroUnitVariance();

       // BaseMultiLayerNetwork network = SerializationUtils.readObject(FileUtils.openInputStream(new File("/home/agibsonccc/models/iris-multithreaded.bin")));

        DataSetIterator list = new SamplingDataSetIterator(next,150,3000);

        Conf c = new Conf();
        c.setFinetuneEpochs(30000);
        c.setFinetuneLearningRate(1e-3f);
        c.setLayerSizes(new int[]{4,3,3});
        c.setnIn(4);
        c.setUseAdaGrad(true);
        //c.setRenderWeightEpochs(1000);
        c.setnOut(3);
        c.setSplit(150);
        c.setFunction(Activations.tanh());
        c.setMultiLayerClazz(DBN.class);
        c.setHiddenUnit(RBM.HiddenUnit.RECTIFIED);
        c.setVisibleUnit(RBM.VisibleUnit.GAUSSIAN);
        c.setUseRegularization(false);
        c.setL2(1e-3f);
        c.setDeepLearningParams(new Object[]{1, 1e-6, 30000});
        c.setMomentum(0.5f);

        ActorNetworkRunner runner = new ActorNetworkRunner("master",list);
        runner.setModelSaver(new DefaultModelSaver(new File("/home/agibsonccc/models/iris-multithreaded.bin")));
        runner.setup(c);
        runner.train();
    }


}
