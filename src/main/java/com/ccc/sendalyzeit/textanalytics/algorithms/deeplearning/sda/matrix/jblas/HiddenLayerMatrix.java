package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas;

import java.io.Serializable;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.jblas.DoubleMatrix;

import com.ccc.sendalyzeit.textanalytics.util.MatrixUtil;

/**
 * Vectorized Hidden Layer
 * @author Adam Gibson
 *
 */
public class HiddenLayerMatrix implements Serializable {

	private static final long serialVersionUID = 915783367350830495L;
	public int N;
	//number of in neurons
	public int n_in;
	//number of out neurons
	public int n_out;
	//weight vector
	public DoubleMatrix W;
	//bias
	public DoubleMatrix b;
	public JDKRandomGenerator rng;
	//input for this layer: note
	//that this input is typically cached whenever
	//any input is processed by this layer
	public DoubleMatrix input;





	/**
	 * 
	 * @param N number of examples
	 * @param n_in number of input neurons
	 * @param n_out number of output neurons
	 * @param W the number of weights (maybe null) a uniform distributed
	 * weight vector with the following formula:
	 * a = 1 / n_in 
	 * the uniform distribution is sampled from -a to a
	 * @param b the bias (maybe null)
	 * 
	 * @param rng the rng to use (maybe null) default: seed with 1234
	 * @param input
	 */
	public HiddenLayerMatrix(int N, int n_in, int n_out, DoubleMatrix W, DoubleMatrix b, JDKRandomGenerator rng,DoubleMatrix input) {
		this.N = N;
		this.n_in = n_in;
		this.n_out = n_out;
		this.input = input;
		if(rng == null) {
			this.rng = new JDKRandomGenerator();
			this.rng.setSeed(1234);
		}
		else 
			this.rng = rng;

		if(W == null) {
			//scaled down weights
			//a value between zero and 1
			//causes strange values to be output
			//also closer to more standard practices
			double a = 1.0 / (double) n_in;
			UniformRealDistribution u = new UniformRealDistribution(-a,a);

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

	
	/**
	 * Binomial sampling of the input
	 * The algorithm is a binomial sampling along the 1st axis (also
	 * known as a column wise mean with the 0 based 1st column being the output)
	 * @param input the input matrix
	 * @return the sampled vector
	 */
	public DoubleMatrix sample_h_given_v(DoubleMatrix input) {
		DoubleMatrix ret = MatrixUtil.binomial(outputMatrix(), 1, rng);
		return ret;
	}
	/**
	 * Sample based on earlier specified input
	 * @return
	 */
	public DoubleMatrix sample_h_given_v() {
		DoubleMatrix ret = MatrixUtil.binomial(outputMatrix(), 1, rng);
		return ret;
	}
}