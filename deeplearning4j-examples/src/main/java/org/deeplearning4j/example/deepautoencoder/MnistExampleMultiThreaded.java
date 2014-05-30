package org.deeplearning4j.example.deepautoencoder;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.iterativereduce.actor.core.DefaultModelSaver;
import org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunner;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.scaleout.conf.Conf;

import java.io.File;
import java.util.Collections;

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
		//batches of 10, 60000 examples total
		DataSetIterator iter = new MnistDataSetIterator(80,60000);

		Conf c = new Conf();
		c.setFinetuneEpochs(10000);
        c.setPretrainEpochs(100);
		c.setFinetuneLearningRate(1e-1);
        c.setPretrainLearningRate(1e-1);
		c.setLayerSizes(new int[]{1000, 500, 250, 28});
		c.setnIn(784);
        c.setDropOut(5e-1);
        c.setHiddenUnitByLayer(Collections.singletonMap(3, RBM.HiddenUnit.GAUSSIAN));
        c.setSplit(100);
        c.setSparsity(1e-1);
		c.setUseAdaGrad(true);
		c.setnOut(10);
		c.setMultiLayerClazz(DBN.class);
		c.setMomentum(9e-1);
        c.setDeepLearningParams(new Object[]{1,1e-1,100});
		ActorNetworkRunner runner = new ActorNetworkRunner("master",iter);
        runner.setModelSaver(new DefaultModelSaver(new File("mnist-example-deepautoencoder.ser")));
		runner.setup(c);
		runner.train();
	}

}
