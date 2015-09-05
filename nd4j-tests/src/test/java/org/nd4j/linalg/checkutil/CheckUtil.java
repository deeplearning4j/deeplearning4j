package org.nd4j.linalg.checkutil;

import java.util.Arrays;

import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;

public class CheckUtil {
	
	/**Check first.mmul(second) using Apache commons math mmul. Float/double matrices only.<br>
	 * Returns true if OK, false otherwise.<br>
	 * Checks each element according to relative error (|a-b|/(|a|+|b|); however absolute error |a-b| must
	 * also exceed minAbsDifference for it to be considered a failure. This is necessary to avoid instability
	 * near 0: i.e., Nd4j mmul might return element of 0.0 (due to underflow on float) while Apache commons math
	 * mmul might be say 1e-30 or something (using doubles). 
	 * Throws exception if matrices can't be multiplied
	 * Checks each element of the result. If
	 * @param first First matrix
	 * @param second Second matrix
	 * @param maxRelativeDifference Maximum relative error
	 * @param minAbsDifference Minimum absolute difference for failure
	 * @return true if OK, false if result incorrect
	 */
	public static boolean checkMmul(INDArray first, INDArray second, double maxRelativeDifference, double minAbsDifference ){
		if(first.size(0) != second.size(1)) throw new IllegalArgumentException("first.rows != second.columns");
		RealMatrix rmFirst = convertToApacheMatrix(first);
		RealMatrix rmSecond = convertToApacheMatrix(second);
		
		INDArray result = first.mmul(second);
		RealMatrix rmResult = rmFirst.multiply(rmSecond);
		
		if(!checkShape(rmResult,result)) return false;
		return checkEntries(rmResult,result,maxRelativeDifference,minAbsDifference);
	}
	
	/**Same as checkMmul, but for matrix addition */
	public static boolean checkAdd(INDArray first, INDArray second, double maxRelativeDifference, double minAbsDifference ){
		RealMatrix rmFirst = convertToApacheMatrix(first);
		RealMatrix rmSecond = convertToApacheMatrix(second);
		
		INDArray result = first.add(second);
		RealMatrix rmResult = rmFirst.add(rmSecond);
		
		if(!checkShape(rmResult,result)) return false;
		return checkEntries(rmResult,result,maxRelativeDifference,minAbsDifference);
	}

	/** Same as checkMmul, but for matrix subtraction */
	public static boolean checkSubtract(INDArray first, INDArray second, double maxRelativeDifference, double minAbsDifference ){
		RealMatrix rmFirst = convertToApacheMatrix(first);
		RealMatrix rmSecond = convertToApacheMatrix(second);
		
		INDArray result = first.sub(second);
		RealMatrix rmResult = rmFirst.subtract(rmSecond);
		
		if(!checkShape(rmResult,result)) return false;
		return checkEntries(rmResult,result,maxRelativeDifference,minAbsDifference);
	}
	
	
	private static boolean checkShape(RealMatrix rmResult, INDArray result){
		int[] outShape = {rmResult.getRowDimension(),rmResult.getColumnDimension()};
		if(!Arrays.equals(outShape, result.shape())){
			System.out.println("Failure on shape: " + Arrays.toString(result.shape()) + ", expected " + Arrays.toString(outShape));
			return false;
		}
		return true;
	}
	
	private static boolean checkEntries(RealMatrix rmResult, INDArray result, double maxRelativeDifference, double minAbsDifference){
		int[] outShape = {rmResult.getRowDimension(),rmResult.getColumnDimension()};
		for( int i=0; i<outShape[0]; i++ ){
			for( int j=0; j<outShape[1]; j++ ){
				double expOut = rmResult.getEntry(i, j);
				double actOut = result.getDouble(i,j);
				if(expOut==0.0 && actOut==0.0) continue;
				double absError = Math.abs(expOut-actOut);
				double relError = absError / (Math.abs(expOut)+Math.abs(actOut));
				if(relError > maxRelativeDifference && absError > minAbsDifference ){
					System.out.println("Failure on value: ("+i+","+j+" exp="+expOut + ", act="+actOut + ", absError="+absError + ", relError="+relError);
					return false;
				}
			}
		}
		return true;
	}
	
	public static RealMatrix convertToApacheMatrix(INDArray matrix){
		if(matrix.rank() != 2) throw new IllegalArgumentException("Input rank is not 2 (not matrix)");
		int[] shape = matrix.shape();
		BlockRealMatrix out = new BlockRealMatrix(shape[0],shape[1]);
		for( int i=0; i<shape[0]; i++ ){
			for( int j=0; j<shape[1]; j++ ){
				double value = matrix.getDouble(i, j);
				out.setEntry(i, j, value);
			}
		}
		return out;
	}
}
