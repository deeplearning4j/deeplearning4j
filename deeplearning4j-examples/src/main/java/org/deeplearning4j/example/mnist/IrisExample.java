package org.deeplearning4j.example.mnist;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.dbn.GaussianRectifiedLinearDBN;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.activation.Activations;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class IrisExample {

	private static Logger log = LoggerFactory.getLogger(IrisExample.class);

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		DataSetIterator irisData = new IrisDataSetIterator(150,150);
		DataSet next = irisData.next();
		next.normalizeZeroMeanZeroUnitVariance();

		int numExamples = next.numExamples();
		log.info("Training on " + numExamples);

		GaussianRectifiedLinearDBN cdbn1 = new GaussianRectifiedLinearDBN.Builder()
		.hiddenLayerSizes(new int[]{4,2,3})
		.normalizeByInputRows(true).numberOfInputs(4).numberOfOutPuts(3)
		.useAdaGrad(true).useHiddenActivationsForwardProp(true).withL2(0.01)
		.useRegularization(false).withActivation(Activations.tanh()).withMomentum(0.1)
		.build();

		cdbn1.pretrain(next.getFirst(), 1, 1e-4, 1000);

		cdbn1.finetune(next.getSecond(), 1e-4, 1000);



		Evaluation eval = new Evaluation();

		DoubleMatrix predicted = cdbn1.predict(next.getFirst());
		eval.eval(next.getSecond(),predicted);



		log.info(eval.stats());






	}

}
