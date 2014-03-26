package org.deeplearning4j.example.mnist;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.dbn.CDBN;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.gradient.multilayer.MultiLayerGradientListener;
import org.deeplearning4j.gradient.multilayer.WeightPlotListener;
import org.deeplearning4j.iterativereduce.actor.multilayer.ActorNetworkRunner;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Test2sAnd4s {

	private static Logger log = LoggerFactory.getLogger(Test2sAnd4s.class);
	
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

		if(args.length >= 1) {
			dbn = SerializationUtils.readObject(new File(args[0]));
		}

	
		
		while(iter.hasNext()) {
			DataSet next = iter.next();
			log.info("Evaluating " + Arrays.toString(next.getFirst().toArray()));
			dbn.feedForward(next.getFirst());
			log.info("Hbias mean " + dbn.getLayers()[0].hBiasMean());
			NeuralNetPlotter plotter = new NeuralNetPlotter();
			plotter.plotNetworkGradient(dbn.getLayers()[0], dbn.getLayers()[0].getGradient(Conf.getDefaultRbmParams()));

		}
		
		
		iter.reset();
		
		Evaluation eval = new Evaluation();

		while(iter.hasNext()) {
			DataSet next = iter.next();
			
			DoubleMatrix predict = dbn.predict(next.getFirst());
			DoubleMatrix labels = next.getSecond();
			eval.eval(labels, predict);
			log.info("Current stats " + eval.stats());
		}

		log.info("Prediction f scores and accuracy");
		log.info(eval.stats());

	}

}
