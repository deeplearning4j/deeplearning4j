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
		DoubleMatrix d = MatrixUtil.xorData(1000, 2000).getFirst();
		RandomGenerator g = new MersenneTwister(123);

		RBM r = new RBM.Builder().withSparsity(0.01).useAdaGrad(true)
				.numberOfVisible(d.columns).numHidden(d.columns / 2).withRandom(g).build();

		r.setInput(d);

		RBMOptimizer opt = new RBMOptimizer(r,0.0001,new Object[]{1,0.0001,1000});
		long time = System.currentTimeMillis();
		if(args.length >= 1 && args[0].equals("normal")) {
			NonZeroStoppingConjugateGradient grad = new NonZeroStoppingConjugateGradient(opt);
			grad.optimize(1000);
		}

		else {
			VectorizedNonZeroStoppingConjugateGradient grad = new VectorizedNonZeroStoppingConjugateGradient(opt);
			grad.optimize(1000);
		}
		long end = System.currentTimeMillis();
		long diff = Math.abs(end - time);
		System.out.println("Took " + diff + " ms");

	}

}
