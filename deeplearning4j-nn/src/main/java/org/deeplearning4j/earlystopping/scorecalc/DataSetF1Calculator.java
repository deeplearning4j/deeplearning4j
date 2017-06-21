package org.deeplearning4j.earlystopping.scorecalc;

import org.deeplearning4j.earlystopping.scorecalc.ScoreCalculator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * Given a DataSetIterator: calculate the total f1 for the model on that data
 * set. Typically used to calculate the f1 on a test set.
 */
public class DataSetF1Calculator implements ScoreCalculator<MultiLayerNetwork> {

	private DataSetIterator dataSetIterator;

	/**
	 * Calculate the score (f1 value) on a given data set (usually a test set)
	 *
	 * @param dataSetIterator
	 *            Data set to calculate the score for
	 */
	public DataSetF1Calculator(DataSetIterator dataSetIterator) {
		this.dataSetIterator = dataSetIterator;
	}

	@Override
	public double calculateScore(MultiLayerNetwork network) {
		dataSetIterator.reset();

		Evaluation evaluation = network.evaluate(dataSetIterator);

		return evaluation.f1() * -1;
	}


	@Override
	public String toString() {
		return "DataSetLossCalculator(" + dataSetIterator + ")";
	}
}