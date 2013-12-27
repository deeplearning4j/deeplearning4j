package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.mnist;

import java.io.IOException;

import org.apache.commons.math3.random.MersenneTwister;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.DataSet;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.fetchers.MnistDataFetcher;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.DenoisingAutoEncoderMatrix;

public class DAMnistTest {
	private static Logger log = LoggerFactory.getLogger(DAMnistTest.class);

	@Test
	public void testMnist() throws IOException {
		MnistDataFetcher fetcher = new MnistDataFetcher();
		fetcher.fetch(1200);
		DataSet data = fetcher.next();

		MersenneTwister rand = new MersenneTwister(123);
		DenoisingAutoEncoderMatrix da = new DenoisingAutoEncoderMatrix.Builder().numberOfVisible(data.getFirst().columns).numHidden(300).withRandom(rand).build();
		double lr = 0.1;
		for(int i = 0; i < 1000; i++) {
			da.train(data.getFirst(), lr, 0.3);
			lr *= 0.95;
			log.info("Cross entropy " + da.getReConstructionCrossEntropy() + " negative log likelihood " + da.negativeLoglikelihood(0.3));

		}

	}

}
