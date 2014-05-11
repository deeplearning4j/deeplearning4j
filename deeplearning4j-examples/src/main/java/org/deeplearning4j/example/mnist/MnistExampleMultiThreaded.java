package org.deeplearning4j.example.mnist;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunner;
import org.deeplearning4j.scaleout.conf.Conf;
/**
 * Equivalent multi threaded example
 * from the {@link MnistExample}
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
		DataSetIterator iter = new MnistDataSetIterator(80,48000);

		Conf c = new Conf();
		c.setFinetuneEpochs(10000);
		c.setFinetuneLearningRate(1e-1);
        c.setPretrainLearningRate(1e-2);
		c.setLayerSizes(new int[]{600,400,250});
		c.setnIn(784);
        c.setUseRegularization(true);
        c.setDropOut(1e-1);
        c.setSparsity(1e-1);
		c.setUseAdaGrad(true);
		c.setnOut(10);
		c.setSplit(10);
		c.setMultiLayerClazz(DBN.class);
		c.setUseRegularization(true);
        c.setL2(2e-2);
		c.setDeepLearningParams(new Object[]{1,1e-1,1000});
		ActorNetworkRunner runner = new ActorNetworkRunner("master",iter);
		runner.setup(c);
		runner.train();
	}

}
