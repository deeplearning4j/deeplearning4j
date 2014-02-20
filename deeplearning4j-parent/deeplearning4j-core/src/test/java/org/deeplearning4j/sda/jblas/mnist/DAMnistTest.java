package org.deeplearning4j.sda.jblas.mnist;

import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.da.DenoisingAutoEncoder;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.datasets.mnist.draw.DrawMnistGreyScale;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class DAMnistTest {
	private static Logger log = LoggerFactory.getLogger(DAMnistTest.class);

	@Test
	public void testMnist() throws Exception {
		MnistDataSetIterator fetcher = new MnistDataSetIterator(500,500);
		MersenneTwister rand = new MersenneTwister(123);

		DenoisingAutoEncoder da = new DenoisingAutoEncoder.Builder().numberOfVisible(784).numHidden(500).withRandom(rand)
				.useRegularization(false)
				.fanIn(0.5).renderWeights(100).withMomentum(0.9).build();

		DataSet first = fetcher.next();
		double error = Double.POSITIVE_INFINITY;

		DoubleMatrix curr = da.getW();
		DoubleMatrix hiddenBias = da.gethBias();
		DoubleMatrix vBias = da.getvBias();
		int numMistakes = 0;

		da.trainTillConvergence(first.getFirst(), 0.1, 0.6);



		log.info(String.valueOf(da.optimizer.getErrors()));

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
			Thread.sleep(1000);
			d.frame.dispose();
			d2.frame.dispose();

		}



	}


}
