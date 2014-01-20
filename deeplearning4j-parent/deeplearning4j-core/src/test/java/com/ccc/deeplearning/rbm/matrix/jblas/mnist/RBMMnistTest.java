package com.ccc.deeplearning.rbm.matrix.jblas.mnist;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.da.DenoisingAutoEncoder;
import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import com.ccc.deeplearning.datasets.mnist.draw.DrawMnistGreyScale;
import com.ccc.deeplearning.plot.NeuralNetPlotter;
import com.ccc.deeplearning.rbm.RBM;
import com.ccc.deeplearning.util.MatrixUtil;

public class RBMMnistTest {
	private static Logger log = LoggerFactory.getLogger(RBMMnistTest.class);

	@Test
	public void testMnist() throws Exception {
		MnistDataSetIterator fetcher = new MnistDataSetIterator(10,10);
		MersenneTwister rand = new MersenneTwister(123);

		DoubleMatrix w = new DoubleMatrix(784,1000);
		w.addi(0.4);
		DataSet first = fetcher.next();

		RBM da = new RBM.Builder().numberOfVisible(784).numHidden(1000).withRandom(rand)
				.fanIn(0.2).withInput(first.getFirst())
				.withMomentum(0.1).build();

		
		
		
		
		da.input = first.getFirst();
		
		NeuralNetPlotter plotter = new NeuralNetPlotter();
		plotter.plot(da);
		
		
		for(int i = 0; i < 1000; i++)
			da.trainTillConvergence(0.1,1,first.getFirst());
		


		DoubleMatrix reconstruct = da.reconstruct(first.getFirst());

		for(int i = 0; i < first.numExamples(); i++) {
			DoubleMatrix draw1 = first.get(i).getFirst().mul(255);
			DoubleMatrix reconstructed2 = reconstruct.getRow(i);
			DoubleMatrix draw2 = MatrixUtil.binomial(reconstructed2,1,new MersenneTwister(123)).mul(255);

			DrawMnistGreyScale d = new DrawMnistGreyScale(draw1);
			d.title = "REAL";
			d.draw();
			DrawMnistGreyScale d2 = new DrawMnistGreyScale(draw2,100,100);
			d2.title = "TEST";
			d2.draw();
			Thread.sleep(10000);
			d.frame.dispose();
			d2.frame.dispose();

		}



	}


}
