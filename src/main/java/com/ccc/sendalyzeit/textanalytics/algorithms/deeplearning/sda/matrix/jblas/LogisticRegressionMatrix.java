package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas;

import java.io.Serializable;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class LogisticRegressionMatrix implements Serializable {
	/**
	 * 
	 */
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
		W = DoubleMatrix.zeros(n_out,n_in);
		b = DoubleMatrix.zeros(n_out);
	}

	public void train(DoubleMatrix x,DoubleMatrix y, final double lr) {
		if(!x.isColumnVector()) {
			log.warn("X " + x.toString() +  " is not a column vector; transforming");
			if(x.isRowVector())
				x = x.transpose();
			else
				throw new IllegalStateException("Unable to transform in to a column vector");
		}

		DoubleMatrix p_y_given_x = softmax(W.mmul(x).add(b));
		final DoubleMatrix dy = y.sub(p_y_given_x);

		for(int i = 0; i < n_out; i++) {
			for(int j = 0; j < n_in; j++) 
				W.put(i,j,W.get(i,j) + lr * dy.get(i) * x.get(j) / N);
			b.put(i,b.get(i) + lr * dy.get(i) / N);
		}


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
		for(int i = 0; i < n_out; i++) {
			ret.put(i,W.getRow(i).dot(x) + b.get(i));
		}
		log.info("Matrix before softmax " + ret);
		return softmax(ret);
	}		
}
