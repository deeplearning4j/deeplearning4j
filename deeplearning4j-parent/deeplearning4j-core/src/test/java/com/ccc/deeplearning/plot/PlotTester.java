package com.ccc.deeplearning.plot;

import java.io.IOException;

import org.apache.commons.math3.random.MersenneTwister;
import org.junit.Test;

import com.ccc.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import com.ccc.deeplearning.rbm.RBM;

public class PlotTester {

	@Test
	public void testPlot() throws IOException {
		MnistDataSetIterator fetcher = new MnistDataSetIterator(10,10);
		MersenneTwister rand = new MersenneTwister(123);

		RBM da = new RBM.Builder().numberOfVisible(784).numHidden(1000).withRandom(rand)
				.withMomentum(0.1).build();

		da.input = fetcher.next().getFirst();
		
		NeuralNetPlotter plotter = new NeuralNetPlotter();
		//plotter.plotWeights(r);
		plotter.plotHbias(da);
		
	}

}
