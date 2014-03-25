package org.deeplearning4j.example.mnist;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.scaleout.conf.Conf;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TestMnistRender {



	private static Logger log = LoggerFactory.getLogger(TestMnistRender.class);

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		DBN dbn =SerializationUtils.readObject(new File(args[0]));

		//batches of 10, 60000 examples total
		DataSetIterator iter = new MnistDataSetIterator(10,60000);

		Evaluation eval = new Evaluation();

		while(iter.hasNext()) {
			DataSet next = iter.next();
			dbn.feedForward(next.getFirst());
			NeuralNetPlotter plotter = new NeuralNetPlotter();
			plotter.plotNetworkGradient(dbn.getLayers()[0], dbn.getLayers()[0].getGradient(Conf.getDefaultRbmParams()));

			log.info("Current stats " + eval.stats());
		}

		log.info("Prediciton f scores and accuracy");
		log.info(eval.stats());

	}

}
