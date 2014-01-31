package com.ccc.deeplearning.nn;

import static com.ccc.deeplearning.util.MatrixUtil.ensureValidOutcomeMatrix;
import static com.ccc.deeplearning.util.MatrixUtil.log;
import static com.ccc.deeplearning.util.MatrixUtil.oneMinus;
import static com.ccc.deeplearning.util.MatrixUtil.sigmoid;
import static com.ccc.deeplearning.util.MatrixUtil.softmax;

import java.io.Serializable;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

/**
 * Logistic regression implementation with jblas.
 * @author Adam Gibson
 *
 */
public class LogisticRegression implements Serializable {

	private static final long serialVersionUID = -7065564817460914364L;
	//number of inputs from final hidden layer
	public int nIn;
	//number of outputs for labeling
	public int nOut;
	//current input and label matrices
	public DoubleMatrix input,labels;
	//weight matrix
	public DoubleMatrix W;
	//bias
	public DoubleMatrix b;
	//weight decay; l2 regularization
	public double l2 = 0.01;
	public boolean useRegularization = true;

	private LogisticRegression() {}

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

	/**
	 * Train with current input and labels 
	 * with the given learning rate
	 * @param lr the learning rate to use
	 */
	public void train(double lr) {
		train(input,labels,lr);
	}


	/**
	 * Train with the given input
	 * and the currently set labels
	 * @param x the input to use
	 * @param lr the learning rate to use
	 */
	public void train(DoubleMatrix x,double lr) {
		train(x,labels,lr);

	}

	/**
	 * Averages the given logistic regression 
	 * from a mini batch in to this one
	 * @param l the logistic regression to average in to this one
	 * @param batchSize  the batch size
	 */
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
		//weight decay
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
		

		this.input = x;
		this.labels = y;

		//DoubleMatrix regularized = W.transpose().mul(l2);
		LogisticRegressionGradient gradient = getGradient(lr);

		W.addi(gradient.getwGradient());
		b.addi(gradient.getbGradient());

	}

	
	
	

	@Override
	protected LogisticRegression clone()  {
		LogisticRegression reg = new LogisticRegression();
		reg.b = b.dup();
		reg.W = W.dup();
		reg.l2 = this.l2;
		reg.labels = this.labels.dup();
		reg.nIn = this.nIn;
		reg.nOut = this.nOut;
		reg.useRegularization = this.useRegularization;
		reg.input = this.input.dup();
		return reg;
	}

	/**
	 * Gets the gradient from one training iteration
	 * @param lr the learning rate to use for training
	 * @return the gradient (bias and weight matrix)
	 */
	public LogisticRegressionGradient getGradient(double lr) {
		//input activation
		DoubleMatrix p_y_given_x = sigmoid(input.mmul(W).addRowVector(b));
		//difference of outputs
		DoubleMatrix dy = labels.sub(p_y_given_x);
		//weight decay
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
