package org.deeplearning4j.plot;

import java.io.IOException;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.NeuralNetworkGradient;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.rbm.RBM;
import org.jblas.DoubleMatrix;
import org.junit.Test;


public class PlotTester {

	@Test
	public void testPlot() throws IOException {
		MnistDataSetIterator fetcher = new MnistDataSetIterator(20,20);
		MersenneTwister rand = new MersenneTwister(123);

		RBM da = new RBM.Builder().numberOfVisible(784).numHidden(400).withRandom(rand).renderWeights(100)
				.useRegularization(false)
				.withMomentum(0).build();

		DoubleMatrix input = fetcher.next().getFirst();
		da.input = input;

		NeuralNetPlotter plotter = new NeuralNetPlotter();
		NeuralNetworkGradient gradient = da.getGradient(new Object[]{1,0.01});

		for(int i = 0; i < 1000; i++) {
			da.trainTillConvergence(0.01, 1, input);
		}




	}

}
