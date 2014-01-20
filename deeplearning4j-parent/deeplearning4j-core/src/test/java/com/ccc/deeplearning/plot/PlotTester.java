package com.ccc.deeplearning.plot;

import org.junit.Test;

import com.ccc.deeplearning.rbm.RBM;

public class PlotTester {

	@Test
	public void testPlot() {
		RBM r = new RBM.Builder()
		.numberOfVisible(10)
		.numHidden(10)
		.build();
		NeuralNetPlotter plotter = new NeuralNetPlotter();
		plotter.plotWeights(r);
		plotter.plotHbias(r);
		
	}

}
