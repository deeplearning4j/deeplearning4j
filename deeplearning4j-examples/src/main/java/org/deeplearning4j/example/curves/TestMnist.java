package org.deeplearning4j.example.curves;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.RawMnistDataSetIterator;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class TestMnist {

	private static Logger log = LoggerFactory.getLogger(TestMnist.class);
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		DBN dbn = SerializationUtils.readObject(new File(args[0]));
		//batches of 10, 60000 examples total
		DataSetIterator iter = new RawMnistDataSetIterator(10,60000);

		Evaluation eval = new Evaluation();

		while(iter.hasNext()) {
			DataSet next = iter.next();
			INDArray predict = dbn.output(next.getFeatureMatrix());
			INDArray labels = next.getLabels();
			eval.eval(labels, predict);
			log.info("Current stats " + eval.stats());
		}

		log.info("Prediction f scores and accuracy");
		log.info(eval.stats());

	}

}
