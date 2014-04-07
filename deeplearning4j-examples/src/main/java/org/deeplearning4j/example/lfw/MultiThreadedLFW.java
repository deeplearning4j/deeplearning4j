package org.deeplearning4j.example.lfw;

import java.io.File;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.RawMnistDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.dbn.GaussianRectifiedLinearDBN;
import org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunner;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.util.SerializationUtils;

public class MultiThreadedLFW {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		//batches of 10, 60000 examples total
		DataSetIterator iter = new LFWDataSetIterator(80,13233,56,56);
		
		
		Conf c = new Conf();
		c.setFinetuneEpochs(10000);
		c.setFinetuneLearningRate(0.01);
		c.setLayerSizes(new int[]{500,400,250});
		c.setnIn(784);
		c.setDropOut(1);
		c.setMomentum(0.5);
		c.setUseAdaGrad(true);
		//c.setRenderWeightEpochs(1000);
		c.setnOut(10);
		c.setSplit(10);
		c.setNumPasses(3);
		c.setMultiLayerClazz(GaussianRectifiedLinearDBN.class);
		c.setUseRegularization(false);
		c.setDeepLearningParams(new Object[]{1,0.01,1000});
		
		if(args.length < 1) {
			ActorNetworkRunner runner = new ActorNetworkRunner("master",iter);
			runner.setup(c);
			runner.train();
		}
		
		else {
			BaseMultiLayerNetwork network = SerializationUtils.readObject(new File(args[0]));
			ActorNetworkRunner runner = new ActorNetworkRunner("master",iter,network);
			runner.setup(c);
			runner.train();
		}
		
	
	}

}
