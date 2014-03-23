package org.deeplearning4j.example.mnist;

import java.io.File;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.dbn.CDBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Finetune2sAnd4s {

	
	private static Logger log = LoggerFactory.getLogger(Finetune2sAnd4s.class);
	
	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		//batches of 10, 60000 examples total
		File f = new File("twoandfours.bin");
		if(!f.exists())
			Create2sAnd4sDataSet.main(null);
		DataSet twosAndFours = DataSet.load(f);
		DataSetIterator iter = new ListDataSetIterator(twosAndFours.asList(),10);


		//784 input (number of columns in mnist, 10 labels (0-9), no regularization
		CDBN dbn = null;

		if(args.length >= 1) {
			dbn = SerializationUtils.readObject(new File(args[0]));
		}

		while(iter.hasNext()) {
			DataSet next = iter.next();
			dbn.finetune(next.getSecond(), 0.01, 1000);
		}

		
		iter.reset();
		
		SerializationUtils.saveObject(dbn, new File("twosandfoursfinetuned.bin"));
		
		Evaluation eval = new Evaluation();

		while(iter.hasNext()) {
			DataSet next = iter.next();
			DoubleMatrix predict = dbn.predict(next.getFirst());
			DoubleMatrix labels = next.getSecond();
			eval.eval(labels, predict);
			log.info("Current stats " + eval.stats());
		}

		log.info("Prediciton f scores and accuracy");
		log.info(eval.stats());

	}

}
