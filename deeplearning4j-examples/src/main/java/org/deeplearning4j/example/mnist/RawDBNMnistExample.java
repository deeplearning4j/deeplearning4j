package org.deeplearning4j.example.mnist;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.RawMnistDataSetIterator;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.models.featuredetectors.rbm.RBM;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RawDBNMnistExample {

	private static Logger log = LoggerFactory.getLogger(RawDBNMnistExample.class);

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		//batches of 10, 60000 examples total
		DataSetIterator iter = new RawMnistDataSetIterator(10,40);

		//784 input (number of columns in mnist, 10 labels (0-9), no regularization
        DBN dbn = new DBN.Builder().withHiddenUnits(RBM.HiddenUnit.RECTIFIED).withVisibleUnits(RBM.VisibleUnit.GAUSSIAN)
				.hiddenLayerSizes(new int[]{500, 400, 250})
				.numberOfInputs(784).numberOfOutPuts(10)
				.build();
		while(iter.hasNext()) {
			DataSet next = iter.next();
			next.normalizeZeroMeanZeroUnitVariance();
			dbn.pretrain(next.getFeatureMatrix(), 1, 0.0001f, 10000);
		}

		iter.reset();




		while(iter.hasNext()) {
			DataSet next = iter.next();
			next.normalizeZeroMeanZeroUnitVariance();

			dbn.setInput(next.getFeatureMatrix());
			dbn.finetune(next.getLabels(), 0.001f, 10000);
		}


		BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream("mnist-dbn.bin"));
		dbn.write(bos);
		bos.flush();
		bos.close();
		log.info("Saved dbn");


		iter.reset();

		Evaluation eval = new Evaluation();

		while(iter.hasNext()) {
			DataSet next = iter.next();
			INDArray predict = dbn.output(next.getFeatureMatrix());
            INDArray labels = next.getLabels();
			eval.eval(labels, predict);
		}

		log.info("Prediciton f scores and accuracy");
		log.info(eval.stats());

	}

}
