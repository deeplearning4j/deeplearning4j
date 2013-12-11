package com.ccc.sendalyzeit.textanalytics.util;

import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class MatrixUtil {

	public static void assertIntMatrix(DoubleMatrix matrix) {
		for(int i = 0; i < matrix.length; i++) {
			int cast = (int) matrix.get(i);
			if(cast != matrix.get(i))
				throw new IllegalArgumentException("Found something that is not an integer at linear index " + i);
		}
	}
	
	public static DoubleMatrix add(DoubleMatrix a,DoubleMatrix b) {
		return a.addi(b);
	}

	public static DoubleMatrix softmax(DoubleMatrix input) {
		input = input.sub(input.max());
		input = MatrixFunctions.exp(input);
		if(input.columns == 1) {
			DoubleMatrix sum = sum(input,0);
			return input.divi(sum);
		}
		else {
			DoubleMatrix sum = sum(input,1).transpose();
			return input.diviColumnVector(sum);

		}
	}
	public static DoubleMatrix mean(DoubleMatrix input,int axis) {
		DoubleMatrix ret = new DoubleMatrix(input.rows,1);
		//column wise
		if(axis == 0) {
			return input.columnMeans();
		}
		//row wise
		else if(axis == 1) {
			return ret.rowMeans();
		}


		return ret;
	}


	public static DoubleMatrix sum(DoubleMatrix input,int axis) {
		DoubleMatrix ret = new DoubleMatrix(input.rows,1);
		//column wise
		if(axis == 0) {
			for(int i = 0; i < input.columns; i++) {
				ret.put(i,input.getColumn(i).sum());
			}
			return ret;
		}
		//row wise
		else if(axis == 1) {
			for(int i = 0; i < input.rows; i++) {
				ret.put(i,input.getRow(i).sum());
			}
			return ret;
		}

		for(int i = 0; i < input.rows; i++)
			ret.put(i,input.getRow(i).sum());
		return ret;
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
		for(int i = 0; i < ret.length; i++) {
			ret.put(i,MathUtils.binomial(rng, n, p.get(i)));
		}
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
		DoubleMatrix denominator = MatrixFunctions.exp(x.neg()).add(1); 
		denominator = DoubleMatrix.ones(denominator.rows,denominator.columns).div(denominator);
		return denominator;
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

	public static DoubleMatrix oneMinus(DoubleMatrix ep) {
		return DoubleMatrix.ones(ep.rows, ep.columns).sub(ep);
	}
	
	public static DoubleMatrix oneDiv(DoubleMatrix ep) {
		for(int i = 0; i < ep.rows; i++) {
			for(int j = 0; j < ep.columns; j++) {
				if(ep.get(i,j) == 0) {
					ep.put(i,j,0.01);
				}
			}
		}
		return DoubleMatrix.ones(ep.rows, ep.columns).div(ep);
	}

}
