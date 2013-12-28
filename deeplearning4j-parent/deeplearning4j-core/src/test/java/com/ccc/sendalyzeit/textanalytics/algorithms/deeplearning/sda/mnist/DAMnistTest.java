package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.mnist;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.DataSet;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.fetchers.MnistDataFetcher;
import com.ccc.sendalyzeit.textanalytics.algorithms.datasets.iterator.impl.MnistDataSetIterator;
import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.DenoisingAutoEncoder;

public class DAMnistTest {
	private static Logger log = LoggerFactory.getLogger(DAMnistTest.class);

	@Test
	public void testMnist() throws IOException {
		MnistDataSetIterator fetcher = new MnistDataSetIterator(300,300);
		MersenneTwister rand = new MersenneTwister(123);

		DenoisingAutoEncoder da = new DenoisingAutoEncoder.Builder().numberOfVisible(784).numHidden(100).withRandom(rand).build();
		DataSet data = null;
		if(fetcher.hasNext()) {
			data = fetcher.next();

			double lr = 0.1;
			da.trainTillConverge(data.getFirst(), lr, 0.6);
			log.info("Cross entropy " + da.getReConstructionCrossEntropy() + " negative log likelihood " + da.negativeLoglikelihood(0.6));

		}

		DoubleMatrix reconstructed = da.reconstruct(data.getFirst());

		for(int i = 0; i < data.getFirst().length; i++) {
			double d = data.getFirst().get(i);
			double r =reconstructed.get(i);
			if(d == 1)
				log.info("D " + d + " R " + r);
		}



	}

}
