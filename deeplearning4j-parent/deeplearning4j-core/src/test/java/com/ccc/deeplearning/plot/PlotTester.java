package com.ccc.deeplearning.plot;

import java.io.IOException;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import com.ccc.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import com.ccc.deeplearning.rbm.RBM;

public class PlotTester {

	@Test
	public void testPlot() throws IOException {
		MnistDataSetIterator fetcher = new MnistDataSetIterator(300,300);
		MersenneTwister rand = new MersenneTwister(123);

		RBM da = new RBM.Builder().numberOfVisible(784).numHidden(500).withRandom(rand)
				.withMomentum(0.1).build();

		DoubleMatrix input = fetcher.next().getFirst();
		da.input = input;
		
		NeuralNetPlotter plotter = new NeuralNetPlotter();

		for(int i = 0; i < 1000; i++) {
			//plotter.plotWeights(r);
			plotter.plotHbias(da);
			da.trainTillConvergence(0.01, 1, input);
		}
		
		
	
		
	}

}
