package org.deeplearning4j.util;

import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.optimize.VectorizedNonZeroStoppingConjugateGradient;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.rbm.RBMOptimizer;
import org.jblas.DoubleMatrix;

public class GradientTesting {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
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

		r.setInput(d);
		
		RBMOptimizer opt = new RBMOptimizer(r,0.01,new Object[]{1,0.01,1000});
		if(args.length >= 1 && args[0].equals("normal")) {
			NonZeroStoppingConjugateGradient grad = new NonZeroStoppingConjugateGradient(opt);
			grad.optimize(3);
		}

		else {
			VectorizedNonZeroStoppingConjugateGradient grad = new VectorizedNonZeroStoppingConjugateGradient(opt);
			grad.optimize(3);
		}
		
	}

}
