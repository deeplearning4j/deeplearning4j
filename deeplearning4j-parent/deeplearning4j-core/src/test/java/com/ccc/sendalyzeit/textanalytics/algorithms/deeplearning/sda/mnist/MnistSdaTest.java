package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.mnist;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.eval.Evaluation;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.DataSet;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator.impl.MnistDataSetIterator;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.StackedDenoisingAutoEncoder;

public class MnistSdaTest extends DeepLearningTest {

	private static Logger log = LoggerFactory.getLogger(MnistSdaTest.class);

	@Test
	public void testMnist() throws IOException {
		MnistDataSetIterator iter = new MnistDataSetIterator(600,6000);
		RandomGenerator rng = new MersenneTwister(123);


		DataSet first = iter.next();
		int numIns = first.getFirst().columns;
		int numLabels = first.getSecond().columns;
		int[] layerSizes = {500,500,2000};


		double lr = 0.1;


		StackedDenoisingAutoEncoder sda = new StackedDenoisingAutoEncoder.Builder().numberOfInputs(numIns)
				.numberOfOutPuts(numLabels).withRng(rng)
				.hiddenLayerSizes(layerSizes).build();


		sda.pretrain(first.getFirst(), lr, 0.6, 100);
		sda.finetune(first.getSecond(), lr,50);

		while(iter.hasNext()) {
			first = iter.next();
			sda.pretrain(first.getFirst(), lr, 0.6, 100);
			sda.finetune(first.getSecond(), lr,50);

		}


		Evaluation eval = new Evaluation();
		log.info("BEGIN EVAL ON " + first.numExamples());
		//	while(iter.hasNext()) {

		DoubleMatrix predicted = sda.predict(first.getFirst());
		log.info("Predicting\n " + predicted.toString().replaceAll(";","\n"));

		eval.eval(first.getSecond(), predicted);
		log.info(eval.stats());
		log.info("Loss is " + sda.negativeLogLikelihood());


		//}




	}

}
