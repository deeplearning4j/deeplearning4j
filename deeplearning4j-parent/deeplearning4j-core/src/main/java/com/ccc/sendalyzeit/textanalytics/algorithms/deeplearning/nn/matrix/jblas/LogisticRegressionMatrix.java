package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas;

import java.io.Serializable;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.ccc.sendalyzeit.textanalytics.util.MatrixUtil;

/**
 * Logistic regression implementation with jblas.
 * @author Adam Gibson
 *
 */
public class LogisticRegressionMatrix implements Serializable {

	private static final long serialVersionUID = -7065564817460914364L;
	public int nIn;
	public int nOut;
	public DoubleMatrix input,labels;
	public DoubleMatrix W;
	public DoubleMatrix b;
	private static Logger log = LoggerFactory.getLogger(LogisticRegressionMatrix.class);



	public LogisticRegressionMatrix(DoubleMatrix input,DoubleMatrix labels, int nIn, int nOut) {
		this.input = input;
		this.labels = labels;
		this.nIn = nIn;
		this.nOut = nOut;
		W = DoubleMatrix.zeros(nIn,nOut);
		b = DoubleMatrix.zeros(nOut);
	}

	public LogisticRegressionMatrix(DoubleMatrix input, int nIn, int nOut) {
		this(input,null,nIn,nOut);
	}


	public void train(double lr) {
		train(input,labels,lr);
	}


	public void train(DoubleMatrix x,double lr) {
		train(x,labels,lr);

	}

	/**
	 * Objective function:  minimize negative log likelihood
	 * @return the negative log likelihood of the model
	 */
	public double negativeLogLikelihood() {
		DoubleMatrix sigAct = MatrixUtil.softmax(input.mmul(W).addRowVector(b));
		DoubleMatrix inner = labels.mul(MatrixFunctions.log(sigAct)).add(MatrixUtil.oneMinus(labels).mul(MatrixFunctions.log(MatrixUtil.oneMinus(sigAct))));
		return - inner.rowSums().mean();
	}
	
	/**
	 * Train on the given inputs and labels.
	 * This will assign the passed in values
	 * as fields to this logistic function for 
	 * caching.
	 * @param x the inputs to train on
	 * @param y the labels to train on
	 * @param lr the learning rate
	 */
	public void train(DoubleMatrix x,DoubleMatrix y, double lr) {
		this.input = x;
		this.labels = y;
		
		if(x.rows != y.rows)
			throw new IllegalArgumentException("Can't train on the 2 given inputs and labels");
		DoubleMatrix mul = x.mmul(W);

		DoubleMatrix p_y_given_x = MatrixUtil.softmax(mul.addRowVector(b));
		DoubleMatrix dy = y.sub(p_y_given_x);
	
		DoubleMatrix mult2 = x.transpose().mmul(dy);
		mult2 = mult2.mul(lr);
		//TECHNICALLY THE CALCULATION COULD INCLUDE L2REG WHICH IS THE FOLLOWING: 
		//lr * x^T * y - lr * L2_reg * W
		//lr * x^T * y is all that is needed; if L2_Reg is zero
		//it will zero out the rest of that quantity
		W = W.add(mult2);
		DoubleMatrix bAdd = dy.columnMeans();
		b = b.add(bAdd.mul(lr));

	}





	/**
	 * Classify input
	 * @param x the input (can either be a matrix or vector)
	 * If it's a matrix, each row is considered an example
	 * and associated rows are classified accordingly.
	 * Each row will be the likelihood of a label given that example
	 * @return a probability distribution for each row
	 */
	public DoubleMatrix predict(DoubleMatrix x) {
		DoubleMatrix ret = x.mmul(W).addRowVector(b);
		return MatrixUtil.softmax(ret);
	}	



	public static class Builder {
		private DoubleMatrix W;
		private LogisticRegressionMatrix ret;
		private DoubleMatrix b;
		private int nIn;
		private int nOut;
		private DoubleMatrix input;


		public Builder withWeights(DoubleMatrix W) {
			this.W = W;
			return this;
		}

		public Builder withBias(DoubleMatrix b) {
			this.b = b;
			return this;
		}

		public Builder numberOfInputs(int nIn) {
			this.nIn = nIn;
			return this;
		}

		public Builder numberOfOutputs(int nOut) {
			this.nOut = nOut;
			return this;
		}

		public LogisticRegressionMatrix build() {
			ret = new LogisticRegressionMatrix(input, nIn, nOut);
			return ret;
		}

	}

}
