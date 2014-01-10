package com.ccc.deeplearning.nn;

import java.io.Serializable;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static com.ccc.deeplearning.util.MatrixUtil.*;

/**
 * Logistic regression implementation with jblas.
 * @author Adam Gibson
 *
 */
public class LogisticRegression implements Serializable {

	private static final long serialVersionUID = -7065564817460914364L;
	public int nIn;
	public int nOut;
	public DoubleMatrix input,labels;
	public DoubleMatrix W;
	public DoubleMatrix b;



	public LogisticRegression(DoubleMatrix input,DoubleMatrix labels, int nIn, int nOut) {
		this.input = input;
		this.labels = labels;
		this.nIn = nIn;
		this.nOut = nOut;
		W = DoubleMatrix.zeros(nIn,nOut);
		b = DoubleMatrix.zeros(nOut);
	}

	public LogisticRegression(DoubleMatrix input, int nIn, int nOut) {
		this(input,null,nIn,nOut);
	}


	public void train(double lr) {
		train(input,labels,lr);
	}


	public void train(DoubleMatrix x,double lr) {
		train(x,labels,lr);

	}
	
	public void merge(LogisticRegression l,int batchSize) {
		W.addi(l.W.subi(W).div(batchSize));
		b.addi(l.b.subi(b).div(batchSize));
	}

	/**
	 * Objective function:  minimize negative log likelihood
	 * @return the negative log likelihood of the model
	 */
	public double negativeLogLikelihood() {
		DoubleMatrix sigAct = softmax(input.mmul(W).addRowVector(b));
		
		return - labels.mul(log(sigAct)).add(
				oneMinus(labels).mul(
						log(oneMinus(sigAct))
				))
				.columnSums().mean();
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
		ensureValidOutcomeMatrix(y);

		this.input = x;
		this.labels = y;

		if(x.rows != y.rows)
			throw new IllegalArgumentException("Can't train on the 2 given inputs and labels");

		DoubleMatrix p_y_given_x = softmax(x.mmul(W).addRowVector(b));
		DoubleMatrix dy = y.sub(p_y_given_x);

		W = W.add(x.transpose().mmul(dy).mul(lr));
		b = b.add(dy.columnMeans().mul(lr));

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
		return softmax(x.mmul(W).addRowVector(b));
	}	



	public static class Builder {
		private DoubleMatrix W;
		private LogisticRegression ret;
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

		public LogisticRegression build() {
			ret = new LogisticRegression(input, nIn, nOut);
			return ret;
		}

	}

}
