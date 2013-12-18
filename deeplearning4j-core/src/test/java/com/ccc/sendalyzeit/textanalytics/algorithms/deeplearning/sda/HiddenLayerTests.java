package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda;

import static org.junit.Assert.*;

import java.util.Arrays;

import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.HiddenLayerMatrix;
import com.ccc.sendalyzeit.textanalytics.util.MatrixUtil;

public class HiddenLayerTests {
	private static Logger log = LoggerFactory.getLogger(HiddenLayerTests.class);
	private RandomGenerator rng;

	@Before
	public void init() {
		rng = new MersenneTwister(1234);
	}

	@Test
	@Ignore
	public void equationTest() {
		double[][] test = new double[][] {
				{1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
				,{1,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
				,{1,1,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0}
				,{1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
				,{0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0}
				,{0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1}
				,{0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1}
				,{0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,0,1}
				,{0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1}
				,{0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0}};
		DoubleMatrix input = new DoubleMatrix(test);

		DoubleMatrix W = new DoubleMatrix(new double[][] {{ -3.08480550e-02, 1.22108771e-02},
				{ -6.22722610e-03, 2.85358584e-02},
				{  2.79975808e-02,  -2.27407395e-02},
				{ -2.23535745e-02, 3.01872178e-02},
				{  4.58139354e-02, 3.75932635e-02},
				{ -1.42182730e-02, 9.95125523e-05},
				{  1.83462935e-02, 2.12702027e-02},
				{ -1.29749245e-02, 6.11961861e-03},
				{  3.08316531e-04,  -4.86231550e-02},
				{  2.72826622e-02, 3.82641191e-02},
				{ -1.35114016e-02, 1.15396178e-02},
				{ -4.24618758e-02 , -1.31175994e-02},
				{  4.33140102e-02, 1.51378143e-02},
				{ -1.02797422e-02, 2.88730143e-02},
				{ -1.83163878e-02, 6.80986526e-03},
				{  3.69127390e-02,  -6.38265761e-03},
				{  3.02147642e-02 , -3.56233175e-02},
				{  2.04260971e-02, 2.04581308e-02},
				{ -2.81207894e-02, 4.24867629e-02},
				{ -5.78592446e-03, 4.09315959e-02}});

		DoubleMatrix testOutput = new DoubleMatrix(new double[][] 
				{{ 0.03312674, 0.10291678},
				{ 0.03935396, 0.07438092},
				{ 0.00482084, 0.17428067},
				{ 0.00153107, 0.065224, },
				{ 0.03669213, 0.05244178},
				{ 0.01239149, 0.11111323},
				{ 0.05485336, 0.12423083},
				{-0.00280173, 0.05348865},
				{-0.00620486, 0.11068602},
				{ 0.03168882, 0.05864201}});



		DoubleMatrix b = new DoubleMatrix(new double[]{0.0,0.0});
		DoubleMatrix preSig = input.mmul(W).addRowVector(b);
		log.info(preSig.toString());

		assertEquals(true,preSig.distance2(testOutput) < 0.01);



		DoubleMatrix postSigmoid = new DoubleMatrix(new double[][]
				{{ 0.50828093, 0.52570651},
				{ 0.50983722, 0.51858666},
				{ 0.50120521, 0.54346022},
				{ 0.50038277, 0.51630022},
				{ 0.509172,  0.51310744},
				{ 0.50309783, 0.52774976},
				{ 0.5137099,  0.53101782},
				{ 0.49929957, 0.51336898},
				{ 0.49844879, 0.52764329},
				{ 0.50792154, 0.5146563 }});

		assertEquals(true,postSigmoid.distance2(MatrixUtil.sigmoid(preSig)) < 0.01);

		DoubleMatrix binomial = new DoubleMatrix(new double[][] {
				{1.0, 1.0},
				{1.0, 0.0},
				{0.0, 1.0},
				{1.0, 0.0},
				{1.0, 1.0},
				{1.0, 0.0},
				{0.0, 1.0},
				{1.0, 0.0},
				{1.0, 0.0},
				{0.0, 0.0}});
		rng.setSeed(1234);

		DoubleMatrix binComp = MatrixUtil.binomial(postSigmoid, 1, rng);

		log.info("Binomial\n " + binomial.toString().replace(";","\n") + "\n comp\n" + binComp.toString().replace(";","\n"));

		assertEquals(true,Arrays.equals(binomial.toArray2(),binComp.toArray2()));
	}

	@Test
	public void testSampling() {
		double[][] test = new double[][] {
				{1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
				,{1,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
				,{1,1,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0}
				,{1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0}
				,{0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0}
				,{0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1}
				,{0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1}
				,{0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,1,1,0,1}
				,{0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1}
				,{0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0}};
		DoubleMatrix input = new DoubleMatrix(test);

		DoubleMatrix W = new DoubleMatrix(new double[][] 
				{{ -3.08480550e-02, 1.22108771e-02},
				{ -6.22722610e-03, 2.85358584e-02},
				{  2.79975808e-02,  -2.27407395e-02},
				{ -2.23535745e-02, 3.01872178e-02},
				{  4.58139354e-02, 3.75932635e-02},
				{ -1.42182730e-02, 9.95125523e-05},
				{  1.83462935e-02, 2.12702027e-02},
				{ -1.29749245e-02, 6.11961861e-03},
				{  3.08316531e-04,  -4.86231550e-02},
				{  2.72826622e-02, 3.82641191e-02},
				{ -1.35114016e-02, 1.15396178e-02},
				{ -4.24618758e-02 , -1.31175994e-02},
				{  4.33140102e-02, 1.51378143e-02},
				{ -1.02797422e-02, 2.88730143e-02},
				{ -1.83163878e-02, 6.80986526e-03},
				{  3.69127390e-02,  -6.38265761e-03},
				{  3.02147642e-02 , -3.56233175e-02},
				{  2.04260971e-02, 2.04581308e-02},
				{ -2.81207894e-02, 4.24867629e-02},
				{ -5.78592446e-03, 4.09315959e-02}});



		DoubleMatrix b = new DoubleMatrix(new double[]{0.0,0.0});





		HiddenLayerMatrix h = new HiddenLayerMatrix(test[0].length, 2, W,b, rng,input);
		log.info(h.sample_h_given_v().toString().replace(";","\n"));
		log.info(h.outputMatrix().toString().replace(";","\n"));

	}

}
