package com.ccc.deeplearning.sda.jblas.mnist;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.base.DeepLearningTest;
import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import com.ccc.deeplearning.eval.Evaluation;
import com.ccc.deeplearning.sda.StackedDenoisingAutoEncoder;

public class MnistSdaTest extends DeepLearningTest {

	private static Logger log = LoggerFactory.getLogger(MnistSdaTest.class);

	@Test
	public void testMnist() throws IOException {
		MnistDataSetIterator iter = new MnistDataSetIterator(600,6000);
		RandomGenerator rng = new MersenneTwister(123);


		DataSet first = iter.next();






		int numIns = first.getFirst().columns;
		int numLabels = first.getSecond().columns;
		int[] layerSizes = {numIns,numIns,numIns};


		double lr = 0.1;


		StackedDenoisingAutoEncoder sda = new StackedDenoisingAutoEncoder.Builder().numberOfInputs(numIns)
				.numberOfOutPuts(numLabels).withRng(rng)
				.hiddenLayerSizes(layerSizes).build();

		DoubleMatrix data1 = first.getFirst().dup();
		DoubleMatrix outcomes = first.getSecond().dup();
		do {
			sda.pretrain(first.getFirst(), lr, 0.3, 1000);
			sda.finetune(outcomes, lr,100);

			Evaluation eval = new Evaluation();
			log.info("BEGIN EVAL ON " + first.numExamples());
			//	while(iter.hasNext()) {

			DoubleMatrix predicted = sda.predict(data1);
			log.info("Predicted\n " + predicted.toString().replaceAll(";","\n"));

			eval.eval(first.getSecond(), predicted);
			log.info(eval.stats());
			log.info("Loss is " + sda.negativeLogLikelihood());
			if(iter.hasNext())
				first = iter.next();
		}while(iter.hasNext());






	}

}
