package com.ccc.sendalyzeit.textanalytics.util;

import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;

public class MatrixUtil {

	public static DoubleMatrix add(DoubleMatrix a,DoubleMatrix b) {
		return a.addi(b);
	}

	/**
	 * Generate a binomial distribution based on the given rng,
	 * a matrix of p values, and a max number.
	 * @param p the p matrix to use
	 * @param n the n to use
	 * @param rng the rng to use
	 * @return a binomial distribution based on the one n, the passed in p values, and rng
	 */
	public static DoubleMatrix binomial(DoubleMatrix p,int n,RandomGenerator rng) {
		DoubleMatrix ret = new DoubleMatrix(p.rows,p.columns);
		for(int i = 0; i < ret.rows; i++)
			for(int j = 0; j < ret.columns; j++)
				ret.put(i,j,MathUtils.binomial(rng, n, p.get(i,j)));
		return ret;
	}
	
	public static DoubleMatrix columnWiseMean(DoubleMatrix x,int axis) {
          DoubleMatrix ret = DoubleMatrix.zeros(x.columns);
          for(int i = 0; i < x.columns; i++) {
        	  ret.put(i,x.getColumn(axis).mean());
          }
          return ret;
	}
	
	public static DoubleMatrix avg(DoubleMatrix...matrices) {
		if(matrices == null)
			return null;
		if(matrices.length == 1)
			return matrices[0];
		else {
			DoubleMatrix ret = matrices[0];
			for(int i = 1; i < matrices.length; i++) 
				ret = ret.add(matrices[i]);
				
			ret = ret.div(matrices.length);
			return ret;
		}
	}
	
	
	public static int maxIndex(DoubleMatrix matrix) {
		double max = matrix.max();
		for(int j = 0; j < matrix.length; j++) {
			if(matrix.get(j) == max)
				return j;
		}
		return -1;
	}
	
	

	public static DoubleMatrix sigmoid(DoubleMatrix x) {
		DoubleMatrix matrix = new DoubleMatrix(x.rows,x.columns);
		for(int i = 0; i < matrix.length; i++)
			matrix.put(i, 1f / (1f + Math.pow(Math.E, -x.get(i))));

		return matrix;
	}
	
	public static DoubleMatrix dot(DoubleMatrix a,DoubleMatrix b) {
		boolean isScalar = a.isColumnVector() || a.isRowVector() && b.isColumnVector() || b.isRowVector();
		if(isScalar) {
			return DoubleMatrix.scalar(a.dot(b));
		}
		else {
			return  a.mmul(b);
		}
	}

	public static DoubleMatrix out(DoubleMatrix a,DoubleMatrix b) {
		return a.mmul(b);
	}

}
