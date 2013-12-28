package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.mnist;

import java.io.IOException;

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
		MnistDataSetIterator iter = new MnistDataSetIterator(100,60000);
		RandomGenerator rng = new MersenneTwister(123);


		DataSet first = iter.next();
		int numIns = first.getFirst().columns;
		int numLabels = first.getSecond().columns;
		int[] layerSizes = {500,500,2000};


		double lr = 0.1;

		StackedDenoisingAutoEncoder sda = new StackedDenoisingAutoEncoder.Builder().numberOfInputs(numIns)
				.numberOfOutPuts(numLabels).withRng(rng)
				.hiddenLayerSizes(layerSizes).build();


		Evaluation eval = new Evaluation();
		while(iter.hasNext()) {
			first = iter.next();
			sda.pretrain(first.getFirst(), lr, 0.6, 50);
			sda.finetune(first.getSecond(), lr,100);



			DoubleMatrix predicted = sda.predict(first.getFirst());
			log.info("Predicting\n " + predicted.toString().replaceAll(";","\n"));

			eval.eval(first.getSecond(), predicted);
			log.info(eval.stats());
			log.info("Loss is " + sda.negativeLogLikelihood());


		}





	}

}
