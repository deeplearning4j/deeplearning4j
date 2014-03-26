package org.deeplearning4j.example.mnist;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.dbn.CDBN;
import org.deeplearning4j.gradient.multilayer.MultiLayerGradientListener;
import org.deeplearning4j.gradient.multilayer.WeightPlotListener;
import org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunner;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.util.SerializationUtils;

public class Classify2sAnd4s {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		//batches of 10, 60000 examples total
		File f = new File("twoandfours.bin");
		if(!f.exists())
			Create2sAnd4sDataSet.main(null);
		DataSet twosAndFours = DataSet.load(f);
		DataSetIterator iter = new ListDataSetIterator(twosAndFours.asList());


		//784 input (number of columns in mnist, 10 labels (0-9), no regularization
		CDBN dbn = null;
		List<MultiLayerGradientListener> listeners = new ArrayList<>();

		WeightPlotListener listener = new WeightPlotListener();
		//listeners.add(listener);


		Conf c = new Conf();
		c.initFromData(twosAndFours);
		c.setFinetuneEpochs(10000);
		c.setFinetuneLearningRate(0.1);
		c.setLayerSizes(new int[]{500,400,250});
		c.setUseAdaGrad(true);
		//c.setRenderWeightEpochs(1000);
		c.setSplit(10);
		c.setNumPasses(3);
		c.setMultiLayerClazz(CDBN.class);
		c.setUseRegularization(false);
		c.setDeepLearningParams(new Object[]{1,0.1,10000});
		//c.setRenderWeightEpochs(1000);
		c.setMultiLayerGradientListeners(listeners);


		if(args.length >= 1) {
			dbn = SerializationUtils.readObject(new File(args[0]));
		}

		ActorNetworkRunner runner = dbn == null ? new ActorNetworkRunner("master",iter) : new ActorNetworkRunner("master",iter,dbn);
		//runner.finetune();
		runner.setup(c);
		runner.train();

	}

}
