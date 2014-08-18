package org.deeplearning4j.example.iris;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.rbm.RBM;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class IrisRBMExample {

	private static Logger log = LoggerFactory.getLogger(IrisRBMExample.class);

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		DataSetIterator irisData = new IrisDataSetIterator(150,150);
		DataSet next = irisData.next();
		next.normalizeZeroMeanZeroUnitVariance();
		
		int numExamples = next.numExamples();
		log.info("Training on " + numExamples);

        RBM r = new RBM.Builder().withHidden(RBM.HiddenUnit.RECTIFIED).withVisible(RBM.VisibleUnit.GAUSSIAN)
		.numberOfVisible(irisData.inputColumns())
		.useAdaGrad(true)
		.numHidden(10).normalizeByInputRows(false).useRegularization(false)
		.build();
		r.trainTillConvergence(next.getFeatureMatrix(),1e-3, new Object[]{1,1e-3,2000});
		log.info("\nData " + String.valueOf("\n" + next.getFeatureMatrix()).replaceAll(";","\n"));
		log.info("\nReconstruct " + String.valueOf("\n" + r.reconstruct(r.getInput())).replaceAll(";","\n"));
		
		

	}


}
