package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.dbn.mnist;

import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.SdAMatrix;

public class MnistSdaTest extends DeepLearningTest {

	private static Logger log = LoggerFactory.getLogger(MnistSdaTest.class);

	@Test
	public void testMnist() throws IOException {
		Pair<DoubleMatrix,DoubleMatrix> first = this.getMnistExample(1);
		int numIns = first.getFirst().columns;
		int numLabels = 10;
		int[] layerSizes = new int[4];
		int numEpochs = 1000;
		Arrays.fill(layerSizes,300);
		double lr = 0.1;
		SdAMatrix sda = new SdAMatrix.Builder().numberOfInputs(numIns)
				.numberOfOutPuts(numLabels).withRng(new MersenneTwister(123))
				.hiddenLayerSizes(layerSizes).build();

		Pair<DoubleMatrix,DoubleMatrix> curr = this.getMnistExampleBatch(1000);
		sda.pretrain(curr.getFirst(), lr, 0.3, numEpochs);
		sda.finetune(curr.getSecond(), lr, numEpochs);

		int numCorrect = 0;

		for(int i = 0; i < 1000; i++) {
			int actualOutcome = SimpleBlas.iamax(curr.getSecond().getRow(i));
			int predicted = SimpleBlas.iamax(sda.predict(curr.getFirst().getRow(i)));
			if(actualOutcome == predicted)
				numCorrect++;
		}

		log.info("Correct " + numCorrect);



	}

}
