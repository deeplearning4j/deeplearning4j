package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas;

import java.io.Serializable;
import java.util.Random;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HiddenLayerMatrix implements Serializable {

	private static final long serialVersionUID = 915783367350830495L;
	public int N;
	public int n_in;
	public int n_out;
	public DoubleMatrix W;
	public DoubleMatrix b;
	public Random rng;
	public DoubleMatrix input;
	private static Logger log = LoggerFactory.getLogger(HiddenLayerMatrix.class);
	public double uniform(double min, double max) {
		return rng.nextDouble() * (max - min) + min;
	}

	public int binomial(int n, double p) {
		if(p < 0 || p > 1) return 0;

		int c = 0;
		double r;

		for(int i = 0; i < n; i++) {
			r = rng.nextDouble();
			if (r < p) 
				c++;
		}

		return c;
	}

	public static double sigmoid(double x) {
		return 1f / (1f + Math.pow(Math.E, -x));
	}

	public static DoubleMatrix sigmoid(DoubleMatrix x) {
		DoubleMatrix matrix = new DoubleMatrix(x.rows,x.columns);
		for(int i = 0; i < matrix.length; i++)
			matrix.put(i, 1f / (1f + Math.pow(Math.E, -x.get(i))));

		return matrix;
	}


	public HiddenLayerMatrix(int N, int n_in, int n_out, DoubleMatrix W, DoubleMatrix b, Random rng) {
		this.N = N;
		this.n_in = n_in;
		this.n_out = n_out;

		if(rng == null)
			this.rng = new Random(1234);
		else this.rng = rng;

		if(W == null) 
			this.W = DoubleMatrix.randn(n_out,n_in);

		else 
			this.W = W;


		if(b == null) 
			this.b = DoubleMatrix.zeros(n_out);
		else 
			this.b = b;
	}

	public DoubleMatrix outputMatrix() {
		return sigmoid(b.add(W.dot(input)));
	}

	public DoubleMatrix outputMatrix(DoubleMatrix input) {
		this.input = input;
		DoubleMatrix ret = new DoubleMatrix(W.rows);
		for(int i = 0; i < W.rows; i++)
			ret.put(i,output(input,W.getRow(i),b.get(i)));
		return ret;
	}

	public double output(DoubleMatrix input, DoubleMatrix w, double b) {
		if(input.length != n_in)
			throw new IllegalArgumentException("Input length must be equal to the number of specified inputs" + n_in);
		if(w.length != n_in)
			throw new IllegalArgumentException("Input length must be equal to the number of specified inputs" + n_in);
		
		
		return sigmoid(w.dot(input) + b);
	}

	public DoubleMatrix outputMatrix(DoubleMatrix input,int i) {
		this.input = input.isColumnVector() ? input : input.transpose();
		DoubleMatrix ret = new DoubleMatrix(1,input.length);
		for(int j = 0 ; j < ret.length; j++) {
			double score = sigmoid(output(input,W.getRow(i),b.get(i)));
			ret.put(j,score);
		}

		return ret;
	}

	public double output() {
		return sigmoid(b.add(W.dot(input)).sum());
	}

	public double output(DoubleMatrix input) {
		this.input = input;
		return sigmoid(b.add(W.dot(input)).sum());
	}

	public double output(DoubleMatrix input,int i) {
		this.input = input;
		return sigmoid(b.get(i) + (W.getRow(i).dot(input)));
	}

	public DoubleMatrix sample_h_given_v(DoubleMatrix input,int length) {
		this.input = input;
		DoubleMatrix ret = DoubleMatrix.zeros(length);
		for(int i = 0; i < ret.length; i++)
			ret.put(i,binomial(1,output(input, W.getRow(i), b.get(i))));
		return ret;
	}
	
	public DoubleMatrix sample_h_given_v(DoubleMatrix input) {
		this.input = input;
		DoubleMatrix ret = DoubleMatrix.zeros(b.length);
		for(int i = 0; i < ret.length; i++)
			ret.put(i,binomial(1,output(input, W.getRow(i), b.get(i))));
		return ret;
	}
	
	public DoubleMatrix sample_h_given_v() {
		DoubleMatrix ret = DoubleMatrix.zeros(input.length);
		for(int i = 0; i < ret.length; i++)
			ret.put(i,binomial(1,output(input, W.getRow(i), b.get(i))));
		return ret;
	}
}
