package com.ccc.deeplearning.nn;

import org.jblas.DoubleMatrix;

public class MatrixTest {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		DoubleMatrix w = new DoubleMatrix(new double[][]{
				{1,3},
				{4,0},
				{2,1},
				
		});
		
		DoubleMatrix row = new DoubleMatrix(new double[][]{
				{1},
				{2},
				{3}
		});

		
		System.out.println(w.addColumnVector(row));
	}

}
