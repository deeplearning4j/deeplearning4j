package org.deeplearning4j.rbm;

import static org.junit.Assert.assertEquals;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.datasets.mnist.draw.DrawMnistGreyScale;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class RBMTest  {

	private static Logger log = LoggerFactory.getLogger(RBMTest.class);


	@Test
	public void testBasic() {
		double[][] data = new double[][]
				{
				{1,1,1,0,0,0},
				{1,0,1,0,0,0},
				{1,1,1,0,0,0},
				{0,0,1,1,1,0},
				{0,0,1,1,0,0},
				{0,0,1,1,1,0},
				{0,0,1,1,1,0}
				};

		DoubleMatrix d = new DoubleMatrix(data);
		RandomGenerator g = new MersenneTwister(123);

		RBM r = new RBM.Builder().withSparsity(0.01)
				.numberOfVisible(6).numHidden(4).withRandom(g).build();
		r.getW().muli(1000);

		r.trainTillConvergence(d,  0.01,new Object[]{1,0.01,1000});

		DoubleMatrix v = new DoubleMatrix(new double[][]
				{{1, 1, 0, 0, 0, 0},
				{0, 0, 0, 1, 1, 0}});	

		log.info("Reconstruction " + r.reconstruct(v).toString());

		NeuralNetwork r2 = r.clone();
		assertEquals(r2.getnVisible(),r.getnVisible());
		assertEquals(r2.getnHidden(),r.getnHidden());
		assertEquals(r2.getW(),r.getW());
		assertEquals(r2.gethBias(),r.gethBias());
		assertEquals(r2.getvBias(),r.getvBias());
		r2.trainTillConvergence(d, 0.01,new Object[]{1,0.01,1000});
		log.info("Cross entropy " + r.getReConstructionCrossEntropy());


	}


	@Test
	public void mnistTest() throws Exception {

		MnistDataFetcher fetcher = new MnistDataFetcher(true);
		fetcher.fetch(10);
		DataSet d = fetcher.next();

		RandomGenerator g = new MersenneTwister(123);

		RBM r = new RBM.Builder()
		.useAdaGrad(true)
		.numberOfVisible(d.numInputs())
		.numHidden(500)
		.withRandom(g)
		.build();
		
		/*for(int i = 0; i < 1000; i++)
			r.train(d.getFirst(), 0.01, new Object[]{1,0.01,3000});
		*/
		/*	for(int i = 0; i < 100; i++)
			r.trainTillConvergence(0.1,1,d.getFirst());
		 */
		
		r.trainTillConvergence(d.getFirst() ,0.01,new Object[]{1,0.01,3000});

		DoubleMatrix reconstruct = r.reconstruct(d.getFirst());
		for(int j = 0; j < d.numExamples(); j++) {

			DoubleMatrix draw1 = d.get(j).getFirst().mul(255);
			DoubleMatrix reconstructed2 = reconstruct.getRow(j);
			DoubleMatrix draw2 = MatrixUtil.binomial(reconstructed2,1,g).mul(255);

			DrawMnistGreyScale d3 = new DrawMnistGreyScale(draw1);
			d3.title = "REAL";
			d3.draw();
			DrawMnistGreyScale d2 = new DrawMnistGreyScale(draw2,1000,1000);
			d2.title = "TEST";
			d2.draw();
			Thread.sleep(10000);
			d3.frame.dispose();
			d2.frame.dispose();
		}


	}



}
