package org.deeplearning4j.example.lfw;

import java.io.File;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.LFWDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunner;
import org.deeplearning4j.iterativereduce.tracker.statetracker.StateTracker;
import org.deeplearning4j.iterativereduce.tracker.statetracker.hazelcast.HazelCastStateTracker;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.scaleout.iterativereduce.multi.UpdateableImpl;
import org.deeplearning4j.util.SerializationUtils;

public class MultiThreadedLFW {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		//batches of 10, 60000 examples total
		DataSetIterator iter = new LFWDataSetIterator(80,13233,28,28);
		
		
		Conf c = new Conf();
		c.setFinetuneEpochs(10000);
		c.setFinetuneLearningRate(1e-2f);
		c.setLayerSizes(new int[]{700,500,250});
		c.setnIn(28 * 28);
		c.setMomentum(0.5f);
        c.setDropOut(1e-1f);
		c.setUseAdaGrad(true);
		//c.setRenderWeightEpochs(1000);
		c.setnOut(10);
		c.setSplit(100);
		c.setNumPasses(1);
        c.setScale(true);
        c.setNormalizeZeroMeanAndUnitVariance(false);
		c.setMultiLayerClazz(DBN.class);
		c.setUseRegularization(true);
        c.setL2(2e-4f);
        c.setHiddenUnit(RBM.HiddenUnit.RECTIFIED);
        c.setVisibleUnit(RBM.VisibleUnit.GAUSSIAN);
		c.setDeepLearningParams(new Object[]{1,1e-2,10000});
        StateTracker<UpdateableImpl> stateTracker = new HazelCastStateTracker();

		if(args.length < 1) {
			ActorNetworkRunner runner = new ActorNetworkRunner("master",iter);
            runner.setStateTracker(stateTracker);
            runner.setup(c);
			runner.train();
		}
		
		else {
			BaseMultiLayerNetwork network = SerializationUtils.readObject(new File(args[0]));
			ActorNetworkRunner runner = new ActorNetworkRunner("master",iter,network);
            runner.setStateTracker(stateTracker);
            runner.setup(c);
			runner.train();
		}
		
	
	}

}
