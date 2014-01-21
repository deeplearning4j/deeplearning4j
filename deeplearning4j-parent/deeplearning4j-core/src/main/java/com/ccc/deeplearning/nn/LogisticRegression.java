package com.ccc.deeplearning.nn;

import static com.ccc.deeplearning.util.MatrixUtil.ensureValidOutcomeMatrix;
import static com.ccc.deeplearning.util.MatrixUtil.log;
import static com.ccc.deeplearning.util.MatrixUtil.oneMinus;
import static com.ccc.deeplearning.util.MatrixUtil.sigmoid;
import static com.ccc.deeplearning.util.MatrixUtil.softmax;

import java.io.Serializable;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
	public double l2 = 0.01;
	public boolean useRegularization = true;
	private static Logger log = LoggerFactory.getLogger(LogisticRegression.class);


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

	public LogisticRegression(int nIn, int nOut) {
		this(null,null,nIn,nOut);
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
		if(useRegularization) {
			double reg = (2 / l2) * MatrixFunctions.pow(this.W,2).sum();
			return - labels.mul(log(sigAct)).add(
					oneMinus(labels).mul(
							log(oneMinus(sigAct))
							))
							.columnSums().mean() + reg;
		}
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
		if(x.rows != y.rows) {
			throw new IllegalArgumentException("How does this happen?");
		}

		this.input = x;
		this.labels = y;

		//DoubleMatrix regularized = W.transpose().mul(l2);
		LogisticRegressionGradient gradient = getGradient(lr);

		W.addi(gradient.getwGradient());
		b.addi(gradient.getbGradient());

	}


	public LogisticRegressionGradient getGradient(double lr) {
		DoubleMatrix p_y_given_x = sigmoid(input.mmul(W).addRowVector(b));
		DoubleMatrix dy = labels.sub(p_y_given_x);
		if(useRegularization)
			dy.divi(input.rows);
		DoubleMatrix wGradient = input.transpose().mmul(dy).mul(lr);
		DoubleMatrix bGradient = dy;
		return new LogisticRegressionGradient(wGradient,bGradient);
		
		
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
			ret.W = W;
			ret.b = b;
			return ret;
		}

	}

}
