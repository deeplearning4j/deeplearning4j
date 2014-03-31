package org.deeplearning4j.rbm;

import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.BaseNeuralNetwork;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
/**
 * RBM with rectified linear hidden units.
 * http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf
 * 
 * @author Adam Gibson
 *
 */
public class RectifiedLinearRBM extends RBM {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8368874372096273122L;


	//never instantiate without the builder
	private RectifiedLinearRBM(){}

	private RectifiedLinearRBM(DoubleMatrix input, int nVisible, int nHidden,
			DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vbias,
			RandomGenerator rng, double fanIn, RealDistribution dist) {
		super(input, nVisible, nHidden, W, hbias, vbias, rng, fanIn, dist);
	}



	@Override
	public Pair<DoubleMatrix, DoubleMatrix> sampleVisibleGivenHidden(DoubleMatrix h) {



		DoubleMatrix v1Mean = propDown(h);
		double diffFromData = MatrixFunctions.pow(input.sub(v1Mean),2).mean() * 1e-4;
		/**
		 * Dynamically set the variance = to the squared 
		 * differences from the mean relative to the data.
		 * 
		 */
		DoubleMatrix gaussianNoise = MatrixUtil.normal(getRng(), v1Mean,diffFromData).mul(diffFromData);

		DoubleMatrix v1Sample = v1Mean.add(gaussianNoise);

		return new Pair<DoubleMatrix,DoubleMatrix>(v1Mean,v1Sample);



	}





	/**
	 * Rectified linear hidden units
	 * @param v the visible values
	 * @return a the hidden samples as rectified linear units
	 */
	public Pair<DoubleMatrix,DoubleMatrix> sampleHiddenGivenVisible(DoubleMatrix v) {
		DoubleMatrix h1Mean = propUp(v);
		//variance wrt reconstruction
		double variance =  MatrixFunctions.pow(v.sub(v.mean()),2).mean();
		/**
		 * Dynamically set the variance = to the squared 
		 * differences from the mean relative to the data.
		 * 
		 */
		
		
		DoubleMatrix gaussianNoise = MatrixUtil.normal(getRng(), MatrixUtil.sigmoid(h1Mean),variance).mul(variance);

		DoubleMatrix h1Sample = h1Mean.add(gaussianNoise);
		if(h1Sample.sum() > 0)
			return new Pair<>(h1Mean,h1Sample);

		else
			return new Pair<>(h1Mean,DoubleMatrix.zeros(h1Sample.rows, h1Sample.columns));

	}


	public static class Builder extends BaseNeuralNetwork.Builder<RectifiedLinearRBM> {
		public Builder() {
			this.clazz = RectifiedLinearRBM.class;
		}
	}


}
