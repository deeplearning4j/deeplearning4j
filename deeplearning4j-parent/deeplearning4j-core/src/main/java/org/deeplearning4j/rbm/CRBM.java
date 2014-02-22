package org.deeplearning4j.rbm;


import static org.deeplearning4j.util.MatrixUtil.log;
import static org.deeplearning4j.util.MatrixUtil.oneDiv;
import static org.deeplearning4j.util.MatrixUtil.oneMinus;
import static org.deeplearning4j.util.MatrixUtil.uniform;
import static org.jblas.MatrixFunctions.exp;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.jblas.DoubleMatrix;


/**
 * Continuous Restricted Boltzmann Machine
 * @author Adam Gibson
 *
 */
public class CRBM extends RBM {

	/**
	 * 
	 */
	private static final long serialVersionUID = 598767790003731193L;


	


	public CRBM() {
		super();
	}

	public CRBM(DoubleMatrix input, int n_visible, int n_hidden,
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias,
			RandomGenerator rng, double fanIn, RealDistribution dist) {
		super(input, n_visible, n_hidden, W, hbias, vbias, rng, fanIn, dist);
	}

	public CRBM(int nVisible, int nHidden, DoubleMatrix W, DoubleMatrix hbias,
			DoubleMatrix vbias, RandomGenerator rng, double fanIn,
			RealDistribution dist) {
		super(nVisible, nHidden, W, hbias, vbias, rng, fanIn, dist);
	}

	@Override
	public DoubleMatrix propDown(DoubleMatrix h) {
		return h.mmul(W.transpose()).addRowVector(vBias);
	}

	@Override
	public Pair<DoubleMatrix, DoubleMatrix> sampleVGivenH(DoubleMatrix h) {
		DoubleMatrix aH = propDown(h);
		DoubleMatrix en = exp(aH.neg());
		DoubleMatrix ep = exp(aH);


		DoubleMatrix v1Mean = oneDiv(oneMinus(en).sub(oneDiv(aH)));
		DoubleMatrix v1Sample = log(
				oneMinus(
				uniform(rng,v1Mean.rows,v1Mean.columns)
				.mul(oneMinus(ep)))
				).div(aH);


		return new Pair<DoubleMatrix,DoubleMatrix>(v1Mean,v1Sample);



	}


	public static class Builder extends BaseNeuralNetwork.Builder<CRBM> {
		public Builder() {
			this.clazz = CRBM.class;
		}
	}



}
