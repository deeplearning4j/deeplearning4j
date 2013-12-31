package com.ccc.deeplearning.sda.jblas.mnist;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.random.MersenneTwister;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.deeplearning.datasets.DataSet;
import com.ccc.deeplearning.datasets.iterator.impl.MnistDataSetIterator;
import com.ccc.deeplearning.datasets.mnist.draw.DrawMnistGreyScale;
import com.ccc.deeplearning.sda.jblas.DenoisingAutoEncoder;
import com.ccc.deeplearning.util.MatrixUtil;

public class DAMnistTest {
	private static Logger log = LoggerFactory.getLogger(DAMnistTest.class);

	@Test
	public void testMnist() throws Exception {
		MnistDataSetIterator fetcher = new MnistDataSetIterator(1000,1000);
		MersenneTwister rand = new MersenneTwister(123);

		DenoisingAutoEncoder da = new DenoisingAutoEncoder.Builder().numberOfVisible(784).numHidden(500).withRandom(rand)
				.withMomentum(0.9).build();


		DataSet first = fetcher.next();
		//for(int i = 0; i < 1000; i++)
		do {
			da.trainTillConverge(first.getFirst(), 0.1, 0.3);
		}while(da.lossFunction(new Object[]{0.3}) > 1);
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
			Thread.sleep(10000);
			d.frame.dispose();
			d2.frame.dispose();

		}



	}


}
