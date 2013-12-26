package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas;

import java.io.Serializable;

import org.jblas.DoubleMatrix;
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

	public void train(DoubleMatrix x,DoubleMatrix y, double lr) {
		if(x.rows != y.rows)
			throw new IllegalArgumentException("Can't train on the 2 given inputs and labels");
		DoubleMatrix mul = x.mmul(W);

		DoubleMatrix p_y_given_x = MatrixUtil.softmax(mul.addRowVector(b));
		DoubleMatrix dy = y.sub(p_y_given_x);
		//self.W += lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
		// self.b += lr * numpy.mean(d_y, axis=0)
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
