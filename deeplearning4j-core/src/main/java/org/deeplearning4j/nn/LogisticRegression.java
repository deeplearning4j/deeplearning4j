package org.deeplearning4j.nn;

import static org.deeplearning4j.util.MatrixUtil.log;
import static org.deeplearning4j.util.MatrixUtil.oneMinus;
import static org.deeplearning4j.util.MatrixUtil.sigmoid;
import static org.deeplearning4j.util.MatrixUtil.softmax;

import java.io.Serializable;

import org.deeplearning4j.nn.gradient.LogisticRegressionGradient;
import org.deeplearning4j.nn.learning.AdaGrad;
import org.deeplearning4j.optimize.LogisticRegressionOptimizer;
import org.deeplearning4j.optimize.VectorizedNonZeroStoppingConjugateGradient;
import org.deeplearning4j.util.MatrixUtil;
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
	private int nIn;
	//number of outputs for labeling
	private int nOut;
	//current input and label matrices
	private DoubleMatrix input,labels;
	//weight matrix
	private DoubleMatrix W;
	//bias
	private DoubleMatrix b;
	//weight decay; l2 regularization
	private double l2 = 0.01;
	private boolean useRegularization = true;
	private boolean useAdaGrad = false;
	private AdaGrad adaGrad;
	private boolean firstTimeThrough;
	private LogisticRegression() {}

	public LogisticRegression(DoubleMatrix input,DoubleMatrix labels, int nIn, int nOut) {
		this.input = input;
		this.labels = labels;
		this.nIn = nIn;
		this.nOut = nOut;
		W = DoubleMatrix.zeros(nIn,nOut);
		adaGrad = new AdaGrad(nIn,nOut);
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
	public  void train(double lr) {
		train(input,labels,lr);
	}


	public void resetAdaGrad(double lr) {
		if(!firstTimeThrough) {
			this.adaGrad = new AdaGrad(nIn,nOut,lr);
			firstTimeThrough = false;
		}
	}

	/**
	 * Train with the given input
	 * and the currently set labels
	 * @param x the input to use
	 * @param lr the learning rate to use
	 */
	public  void train(DoubleMatrix x,double lr) {
		MatrixUtil.complainAboutMissMatchedMatrices(x, labels);

		train(x,labels,lr);

	}

	/**
	 * Run conjugate gradient with the given x and y
	 * @param x the input to use
	 * @param y the labels to use
	 * @param learningRate
	 * @param epochs
	 */
	public  void trainTillConvergence(DoubleMatrix x,DoubleMatrix y, double learningRate,int epochs) {
		MatrixUtil.complainAboutMissMatchedMatrices(x, y);

		this.input = x;
		this.labels = y;
		trainTillConvergence(learningRate,epochs);

	}

	/**
	 * Run conjugate gradient
	 * @param learningRate the learning rate to train with
	 * @param numEpochs the number of epochs
	 */
	public  void trainTillConvergence(double learningRate, int numEpochs) {
		LogisticRegressionOptimizer opt = new LogisticRegressionOptimizer(this, learningRate);
		VectorizedNonZeroStoppingConjugateGradient g = new VectorizedNonZeroStoppingConjugateGradient(opt);
		g.optimize(numEpochs);

	}


	/**
	 * Averages the given logistic regression 
	 * from a mini batch in to this one
	 * @param l the logistic regression to average in to this one
	 * @param batchSize  the batch size
	 */
	public void merge(LogisticRegression l,int batchSize) {
		if(this.useRegularization) {

			W.addi(l.W.subi(W).div(batchSize));
			b.addi(l.b.subi(b).div(batchSize));
		}

		else {
			W.addi(l.W.subi(W));
			b.addi(l.b.subi(b));
		}

	}

	/**
	 * Objective function:  minimize negative log likelihood
	 * @return the negative log likelihood of the model
	 */
	public  double negativeLogLikelihood() {
		MatrixUtil.complainAboutMissMatchedMatrices(input, labels);
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

		else {
			return - labels.mul(log(sigAct)).add(
					oneMinus(labels).mul(
							log(oneMinus(sigAct))
							))
							.columnSums().mean();


		}

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
	public  void train(DoubleMatrix x,DoubleMatrix y, double lr) {
		MatrixUtil.complainAboutMissMatchedMatrices(x, y);

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
		if(this.labels != null)
			reg.labels = this.labels.dup();
		reg.nIn = this.nIn;
		reg.nOut = this.nOut;
		reg.useRegularization = this.useRegularization;
		if(this.input != null)
			reg.input = this.input.dup();
		return reg;
	}

	/**
	 * Gets the gradient from one training iteration
	 * @param lr the learning rate to use for training
	 * @return the gradient (bias and weight matrix)
	 */
	public  LogisticRegressionGradient getGradient(double lr) {
		MatrixUtil.complainAboutMissMatchedMatrices(input, labels);

		//input activation
		DoubleMatrix p_y_given_x = sigmoid(input.mmul(W).addRowVector(b));
		//difference of outputs
		DoubleMatrix dy = labels.sub(p_y_given_x);
		//weight decay
		dy.divi(input.rows);
	
		DoubleMatrix wGradient = input.transpose().mmul(dy);
		if(useAdaGrad)
			wGradient.muli(adaGrad.getLearningRates(wGradient));
		else
			wGradient.muli(lr);

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
	public  DoubleMatrix predict(DoubleMatrix x) {
		this.input = x;
		return softmax(x.mmul(W).addRowVector(b));
	}	



	public  int getnIn() {
		return nIn;
	}

	public  void setnIn(int nIn) {
		this.nIn = nIn;
	}

	public  int getnOut() {
		return nOut;
	}

	public  void setnOut(int nOut) {
		this.nOut = nOut;
	}

	public  DoubleMatrix getInput() {
		return input;
	}

	public  void setInput(DoubleMatrix input) {
		this.input = input;
	}

	public  DoubleMatrix getLabels() {
		return labels;
	}

	public  void setLabels(DoubleMatrix labels) {
		this.labels = labels;
	}

	public  DoubleMatrix getW() {
		return W;
	}

	public  void setW(DoubleMatrix w) {
		W = w;
	}

	public  DoubleMatrix getB() {
		return b;
	}

	public  void setB(DoubleMatrix b) {
		this.b = b;
	}

	public  double getL2() {
		return l2;
	}

	public  void setL2(double l2) {
		this.l2 = l2;
	}

	public  boolean isUseRegularization() {
		return useRegularization;
	}

	public  void setUseRegularization(boolean useRegularization) {
		this.useRegularization = useRegularization;
	}



	public static class Builder {
		private DoubleMatrix W;
		private LogisticRegression ret;
		private DoubleMatrix b;
		private double l2;
		private int nIn;
		private int nOut;
		private DoubleMatrix input;
		private boolean useRegualarization;
		private boolean useAdaGrad = false;

		public Builder useAdaGrad(boolean useAdaGrad) {
			this.useAdaGrad = useAdaGrad; 
			return this;
		}


		public Builder withL2(double l2) {
			this.l2 = l2;
			return this;
		}

		public Builder useRegularization(boolean regularize) {
			this.useRegualarization = regularize;
			return this;
		}

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
			if(W != null)
				ret.W = W;
			if(b != null)
				ret.b = b;
			ret.useRegularization = useRegualarization;
			ret.l2 = l2;
			ret.useAdaGrad = useAdaGrad;

			return ret;
		}

	}

}
