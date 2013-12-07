package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas;

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
	public int N;
	public int n_in;
	public int n_out;
	public DoubleMatrix W;
	public DoubleMatrix b;
	private static Logger log = LoggerFactory.getLogger(LogisticRegressionMatrix.class);

	public LogisticRegressionMatrix(int N, int n_in, int n_out) {
		this.N = N;
		this.n_in = n_in;
		this.n_out = n_out;
		W = DoubleMatrix.zeros(n_in,n_out);
		b = DoubleMatrix.zeros(n_out);
	}

	public void train(DoubleMatrix x,DoubleMatrix y, final double lr) {
		DoubleMatrix mul = x.mmul(W);

		DoubleMatrix p_y_given_x = softmax(mul.addRowVector(b));
		DoubleMatrix dy = y.sub(p_y_given_x);
		//self.W += lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
		// self.b += lr * numpy.mean(d_y, axis=0)
		DoubleMatrix mult2 = x.transpose().mmul(dy);
		//TECHNICALLY THE CALCULATION COULD INCLUDE L2REG WHICH IS THE FOLLOWING: 
		//lr * numpy.dot(self.x.T, d_y) - lr * L2_reg * self.W
		//lr * x.T * d_y is all that is needed; if L2_Reg is zero
		//it will zero out the rest of that quantity
		W = W.add(mult2);
		b = b.add(MatrixUtil.columnWiseMean(p_y_given_x, 0).mul(lr));
		
	}




public DoubleMatrix softmax(DoubleMatrix x) {
	final double max = x.max();
	for(int i = 0; i < x.length; i++)
		x.put(i,Math.exp(x.get(i) - max));
	double sum = x.sum();
	for(int i = 0; i < x.length; i++)
		x.put(i,x.get(i) / sum);

	return x;
}

public DoubleMatrix predict(DoubleMatrix x) {
	DoubleMatrix ret = new DoubleMatrix(1,n_out);
	for(int i = 0; i < n_out; i++) 
		ret.put(i,W.getRow(i).dot(x) + b.get(i));

	return softmax(ret);
}		
}
