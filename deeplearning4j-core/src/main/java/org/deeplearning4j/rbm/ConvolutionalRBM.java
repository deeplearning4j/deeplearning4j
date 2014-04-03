package org.deeplearning4j.rbm;

import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.RangeUtils;
import org.jblas.ranges.Range;

import static org.jblas.MatrixFunctions.*;
import static org.jblas.DoubleMatrix.zeros;

public class ConvolutionalRBM extends RBM {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6868729665328916878L;
	private int numFilters;
	private int poolRows;
	private int poolColumns;




	public DoubleMatrix visibleExpectation(DoubleMatrix visible,double bias) {
		DoubleMatrix filterMatrix = new DoubleMatrix(numFilters);
		for(int k = 0; k < numFilters; k++) {
			DoubleMatrix next = MatrixUtil.convolution2D(visible, 
					visible.columns, 
					visible.rows, this.getW().getRow(k), this.getW().rows, this.getW().columns).add(this.getvBias().add(bias)).transpose();
			filterMatrix.putRow(k,next);
		}

		filterMatrix = pool(filterMatrix);
		
		filterMatrix.addi(1);
		filterMatrix = MatrixUtil.oneDiv(filterMatrix);
		
		//replace with actual function later, sigmoid is only one possible option
		return MatrixUtil.sigmoid(filterMatrix);

	}

	
	
	
	
	public DoubleMatrix pooledExpectation(DoubleMatrix visible,double bias) {
		DoubleMatrix filterMatrix = new DoubleMatrix(numFilters);
		for(int k = 0; k < numFilters; k++) {
			DoubleMatrix next = MatrixUtil.convolution2D(visible, 
					visible.columns, 
					visible.rows, this.getW().getRow(k), this.getW().rows, this.getW().columns).add(this.gethBias().add(bias)).transpose();
			filterMatrix.putRow(k,next);
		}

		filterMatrix = pool(filterMatrix);
		
		filterMatrix.addi(1);
		filterMatrix = MatrixUtil.oneDiv(filterMatrix);
		
		
		return filterMatrix;

	}

	public DoubleMatrix pool(DoubleMatrix hidden) {
		DoubleMatrix active = exp(hidden.transpose());
		DoubleMatrix pool = zeros(active.rows,active.columns);
		int maxColumn = (int) Math.ceil(poolColumns/ hidden.columns);
		for(int j = 0;j < maxColumn; j++) {
			int beginColumnSlice = j * poolColumns;
			int endColumnSlice = (j + 1) * poolColumns;
			int maxRow = (int) Math.ceil(poolRows / hidden.rows);
			for(int i = 0; i < maxRow; i++) {
				int beginRowSlice = i * poolRows;
				int endRowSlice = (i + 1) * poolRows;
				DoubleMatrix subSlice = active.get(new int[]{beginRowSlice,endRowSlice},
						new int[]{beginColumnSlice,endColumnSlice})
						.rowSums()
						.rowSums();
				pool.put(new int[]{beginRowSlice,endRowSlice},
						new int[]{beginColumnSlice,endColumnSlice}, subSlice);
			}

		}
		return pool.transpose();

	}



}
