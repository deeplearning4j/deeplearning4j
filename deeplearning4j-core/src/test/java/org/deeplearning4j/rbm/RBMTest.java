package org.deeplearning4j.rbm;

import static org.junit.Assert.assertEquals;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.dataset.DataSet;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.jblas.NDArray;
import org.deeplearning4j.linalg.jblas.util.JblasSerde;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.WeightInit;
import org.jblas.DoubleMatrix;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.DataInputStream;


public class RBMTest  {

	private static Logger log = LoggerFactory.getLogger(RBMTest.class);


	@Test
	public void testBasic() throws Exception {
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




        ClassPathResource r3 = new ClassPathResource("/test-matrix.ser");


        NDArrays.factory().setOrder('f');


		INDArray d =  NDArrays.create(data);

        RandomGenerator g = new MersenneTwister(123);

		RBM r = new RBM.Builder().weightInit(WeightInit.SI)
				.numberOfVisible(d.columns()).numHidden(4).withRandom(g).build();



		r.trainTillConvergence(d,  0.01f,new Object[]{1,0.01f,1000});

        double d5 = r.getReConstructionCrossEntropy();

		INDArray v = new NDArray(new float[]
				{1, 1, 0, 0, 0, 0,0, 0, 0, 1, 1, 0}, new int[]{2,6});

		log.info("Reconstruction " + r.reconstruct(v).toString());

		NeuralNetwork r2 = r.clone();
		assertEquals(r2.getnVisible(),r.getnVisible());
		assertEquals(r2.getnHidden(),r.getnHidden());
		assertEquals(r2.getW(),r.getW());
		assertEquals(r2.gethBias(),r.gethBias());
		assertEquals(r2.getvBias(),r.getvBias());
		r2.trainTillConvergence(d, 0.01f,new Object[]{1,0.01f,1000});


	}


	@Test
	public void mnistTest() throws Exception {

		MnistDataFetcher fetcher = new MnistDataFetcher(true);
		fetcher.fetch(10);
		DataSet d = fetcher.next();

		RandomGenerator g = new MersenneTwister(123);

		RBM r = new RBM.Builder()
		.numberOfVisible(d.numInputs())
		.numHidden(600)
		.build();
		

		
		r.trainTillConvergence(d.getFeatureMatrix() ,1e-2f,new Object[]{1,1e-1f,100});




	}



}
