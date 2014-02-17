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
		MnistDataSetIterator fetcher = new MnistDataSetIterator(100,100);
		MersenneTwister rand = new MersenneTwister(123);

		
		DataSet first = fetcher.next();

		RBM da = new RBM.Builder().numberOfVisible(784).numHidden(400).withRandom(rand)
				.useRegularization(false)
				.withMomentum(0).build();




		da.input = first.getFirst();

		da.trainTillConvergence(first.getFirst(), 0.01, new Object[]{1,0.01,1000});
		DoubleMatrix reconstruct = da.reconstruct(first.getFirst());

		for(int j = 0; j < first.numExamples(); j++) {
			DoubleMatrix draw1 = first.get(j).getFirst().mul(255);
			DoubleMatrix reconstructed2 = reconstruct.getRow(j);
			DoubleMatrix draw2 = MatrixUtil.binomial(reconstructed2,1,new MersenneTwister(123)).mul(255);

			DrawMnistGreyScale d = new DrawMnistGreyScale(draw1);
			d.title = "REAL";
			d.draw();
			DrawMnistGreyScale d2 = new DrawMnistGreyScale(draw2,100,100);
			d2.title = "TEST";
			d2.draw();
			Thread.sleep(1000);
			d.frame.dispose();
			d2.frame.dispose();

		}
		
		for(int i = 0; i < 3000; i++) {
			if(i% 500 == 0 || i == 0) {
				reconstruct = da.reconstruct(first.getFirst());
				if(i > 0)
					for(int j = 0; j < first.numExamples(); j++) {
						DoubleMatrix draw1 = first.get(j).getFirst().mul(255);
						DoubleMatrix reconstructed2 = reconstruct.getRow(j);
						DoubleMatrix draw2 = MatrixUtil.binomial(reconstructed2,1,new MersenneTwister(123)).mul(255);

						DrawMnistGreyScale d = new DrawMnistGreyScale(draw1);
						d.title = "REAL";
						d.draw();
						DrawMnistGreyScale d2 = new DrawMnistGreyScale(draw2,100,100);
						d2.title = "TEST";
						d2.draw();
						Thread.sleep(1000);
						d.frame.dispose();
						d2.frame.dispose();

					}
			}
			da.train(first.getFirst(), 0.01, new Object[]{1});
			log.info("Negative log likelihood " + da.getReConstructionCrossEntropy());


		}







	}


}
