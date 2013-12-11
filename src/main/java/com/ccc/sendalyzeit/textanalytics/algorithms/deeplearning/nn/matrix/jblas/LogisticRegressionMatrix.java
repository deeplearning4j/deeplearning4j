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
	public int n_in;
	public int n_out;
	public DoubleMatrix input;
	public DoubleMatrix W;
	public DoubleMatrix b;
	private static Logger log = LoggerFactory.getLogger(LogisticRegressionMatrix.class);

	public LogisticRegressionMatrix(DoubleMatrix input, int n_in, int n_out) {
		this.input = input;
		this.n_in = n_in;
		this.n_out = n_out;
		W = DoubleMatrix.zeros(n_in,n_out);
		b = DoubleMatrix.zeros(n_out);
	}

	public void train(DoubleMatrix x,DoubleMatrix y, double lr) {
		DoubleMatrix mul = x.mmul(W);

		DoubleMatrix p_y_given_x = MatrixUtil.softmax(mul.addRowVector(b));
		DoubleMatrix dy = y.sub(p_y_given_x);
		//self.W += lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
		// self.b += lr * numpy.mean(d_y, axis=0)
		DoubleMatrix mult2 = x.transpose().mmul(dy);
		mult2 = mult2.mul(lr);
		//TECHNICALLY THE CALCULATION COULD INCLUDE L2REG WHICH IS THE FOLLOWING: 
		//lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
		//lr * x.T * d_y is all that is needed; if L2_Reg is zero
		//it will zero out the rest of that quantity
		W = W.add(mult2);
		DoubleMatrix bAdd = dy.columnMeans();
		b = b.add(bAdd.mul(lr));

	}




	

	public DoubleMatrix predict(DoubleMatrix x) {
		DoubleMatrix ret = x.mmul(W).addRowVector(b);
		return MatrixUtil.softmax(ret);
	}		
}
