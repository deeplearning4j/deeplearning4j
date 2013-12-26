package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.mnist;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.jblas.SimpleBlas;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;
import com.ccc.sendalyzeit.deeplearning.eval.Evaluation;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.fetchers.MnistDataFetcher;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.base.DeepLearningTest;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.SdAMatrix;

public class MnistSdaTest extends DeepLearningTest {

	private static Logger log = LoggerFactory.getLogger(MnistSdaTest.class);

	@Test
	public void testMnist() throws IOException {
		MnistDataFetcher fetcher = new MnistDataFetcher();
		fetcher.fetch(1200);
		Pair<DoubleMatrix,DoubleMatrix> first = fetcher.next();
		int numIns = first.getFirst().columns;
		int numLabels = first.getSecond().columns;
		int[] layerSizes = new int[3];
		Arrays.fill(layerSizes,100);
		double lr = 0.1;
		SdAMatrix sda = new SdAMatrix.Builder().numberOfInputs(numIns)
				.numberOfOutPuts(numLabels).withRng(new MersenneTwister(123))
				.hiddenLayerSizes(layerSizes).build();

		sda.pretrain(first.getFirst(), lr, 0.5, 200);
		sda.finetune(first.getSecond(), lr,200);





		DoubleMatrix predicted = sda.predict(first.getFirst());
		//log.info("Predicting\n " + first.getFirst().toString().replaceAll(";","\n"));

		Evaluation eval = new Evaluation();
		eval.eval(first.getSecond(), predicted);
		log.info(eval.stats());
	}

}
