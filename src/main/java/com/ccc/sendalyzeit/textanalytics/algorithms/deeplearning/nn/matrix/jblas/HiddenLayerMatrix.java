package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas;

import java.io.Serializable;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;

import com.ccc.sendalyzeit.textanalytics.util.MatrixUtil;

/**
 * Vectorized Hidden Layer
 * @author Adam Gibson
 *
 */
public class HiddenLayerMatrix implements Serializable {

	private static final long serialVersionUID = 915783367350830495L;
	public int n_in;
	public int n_out;
	public DoubleMatrix W;
	public DoubleMatrix b;
	public RandomGenerator rng;
	public DoubleMatrix input;





	public HiddenLayerMatrix(int n_in, int n_out, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input) {
		this.n_in = n_in;
		this.n_out = n_out;
		this.input = input;

		if(rng == null) {
			this.rng = new MersenneTwister(1234);
		}
		else 
			this.rng = rng;

		if(W == null) {
			double a = 1.0 / (double) n_in;

			UniformRealDistribution u = new UniformRealDistribution(rng,-a,a);

			this.W = DoubleMatrix.zeros(n_in,n_out);

			for(int i = 0; i < this.W.rows; i++) 
				for(int j = 0; j < this.W.columns; j++) 
					this.W.put(i,j,u.sample());
		}

		else 
			this.W = W;


		if(b == null) 
			this.b = DoubleMatrix.zeros(n_out);
		else 
			this.b = b;
	}

	public DoubleMatrix outputMatrix() {
		DoubleMatrix mult = this.input.mmul(W);
		mult = mult.addRowVector(b);
		return MatrixUtil.sigmoid(mult);
	}

	public DoubleMatrix outputMatrix(DoubleMatrix input) {
		this.input = input;
		return outputMatrix();
	}









	public DoubleMatrix sample_h_given_v(DoubleMatrix input) {
		this.input = input;
		//reset the seed to ensure consistent generation of data
		DoubleMatrix ret = MatrixUtil.binomial(outputMatrix(), 1, rng);
		return ret;
	}

	public DoubleMatrix sample_h_given_v() {
		DoubleMatrix output = outputMatrix();
		//reset the seed to ensure consistent generation of data
		DoubleMatrix ret = MatrixUtil.binomial(output, 1, rng);
		return ret;
	}
}