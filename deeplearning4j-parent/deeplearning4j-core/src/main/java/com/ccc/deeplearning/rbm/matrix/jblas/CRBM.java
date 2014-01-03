package com.ccc.deeplearning.rbm.matrix.jblas;

import static com.ccc.deeplearning.util.MatrixUtil.log;
import static com.ccc.deeplearning.util.MatrixUtil.oneDiv;
import static com.ccc.deeplearning.util.MatrixUtil.oneMinus;
import static com.ccc.deeplearning.util.MatrixUtil.uniform;

import static org.jblas.MatrixFunctions.exp;

import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;

import com.ccc.deeplearning.berkeley.Pair;
import com.ccc.deeplearning.nn.matrix.jblas.BaseNeuralNetwork;

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


	public CRBM() {}



	public CRBM(DoubleMatrix input, int nVisible, int nHidden, DoubleMatrix W,
			DoubleMatrix hBias, DoubleMatrix vBias, RandomGenerator rng) {
		super(input, nVisible, nHidden, W, hBias, vBias, rng);
	}



	public CRBM(int n_visible, int n_hidden, DoubleMatrix W,
			DoubleMatrix hbias, DoubleMatrix vbias, RandomGenerator rng) {
		super(n_visible, n_hidden, W, hbias, vbias, rng);
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
