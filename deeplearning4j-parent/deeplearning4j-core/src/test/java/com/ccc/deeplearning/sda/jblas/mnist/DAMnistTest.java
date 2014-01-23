package com.ccc.deeplearning.sda.jblas.mnist;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.da.DenoisingAutoEncoder;
import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import com.ccc.deeplearning.datasets.mnist.draw.DrawMnistGreyScale;
import com.ccc.deeplearning.util.MatrixUtil;

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

		da.trainTillConverge(first.getFirst(), 0.1, 0.6);



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
