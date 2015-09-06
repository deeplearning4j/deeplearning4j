package org.nd4j.linalg.checkutil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**@author Alex Black
 */
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
		if(first.size(1) != second.size(0)) throw new IllegalArgumentException("first.columns != second.rows");
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
		for( int i = 0; i < outShape[0]; i++) {
			for( int j = 0; j < outShape[1]; j++) {
				double expOut = rmResult.getEntry(i, j);
				double actOut = result.getDouble(i,j);
				if(expOut==0.0 && actOut==0.0)
                    continue;
				double absError = Math.abs(expOut - actOut);
				double relError = absError / (Math.abs(expOut) + Math.abs(actOut));
				if(relError > maxRelativeDifference && absError > minAbsDifference) {
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

	/** Get an array of INDArrays (2d) all with the specified shape. Each NDArray in the
	 * returned array has been obtained by applying an operation such as transpose, tensorAlongDimension,
	 * etc to an original array.
	 */
	public static INDArray[] getAllTestMatricesWithShape(int rows, int cols) {
		List<INDArray> all = new ArrayList<>();
		all.add(getTransposedMatrixWithShape(rows,cols));

		INDArray[] temp = getSubMatricesWithShape(rows,cols);
		for( int i = 0; i<temp.length; i++ )
			all.add(temp[i]);

		temp = getTensorAlongDimensionMatricesWithShape(rows, cols);
		for(int i = 0; i < temp.length; i++ )
			all.add(temp[i]);

		all.add(getPermutedWithShape(rows,cols));
		all.add(getReshapedWithShape(rows,cols));

		return all.toArray(new INDArray[all.size()]);
	}

	public static INDArray getTransposedMatrixWithShape(int row, int col){
		INDArray out = Nd4j.rand(new int[]{col,row});
		return out.transpose();
	}

	public static INDArray[] getSubMatricesWithShape(int rows, int cols){
		INDArray orig = Nd4j.rand(new int[]{2*rows,2*cols});
		INDArray first = orig.get(NDArrayIndex.interval(0, rows),NDArrayIndex.interval(0, cols));
		INDArray second = orig.get(NDArrayIndex.interval(3, rows+3),NDArrayIndex.interval(3,cols+3));
		INDArray third = orig.get(NDArrayIndex.interval(rows,2*rows),NDArrayIndex.interval(cols,2*cols));
		return new INDArray[]{first,second,third};
	}

	public static INDArray[] getTensorAlongDimensionMatricesWithShape(int rows, int cols){
		//From 3d NDArray: do various tensors. One offset 0, one offset > 0
		//[0,1], [0,2], [1,0], [1,2], [2,0], [2,1]
		INDArray[] out = new INDArray[12];
		INDArray temp01 = Nd4j.rand(new int[]{cols,rows,4});
		out[0] = temp01.tensorAlongDimension(0, 0,1);
		out[1] = temp01.tensorAlongDimension(2, 0,1);

		INDArray temp02 = Nd4j.rand(new int[]{cols,4,rows});
		out[2] = temp02.tensorAlongDimension(0, 0,2);
		out[3] = temp02.tensorAlongDimension(2, 0,2);

		INDArray temp10 = Nd4j.rand(new int[]{rows,cols,4});
		out[4] = temp10.tensorAlongDimension(0, 1,0);
		out[5] = temp10.tensorAlongDimension(2, 1,0);

		INDArray temp12 = Nd4j.rand(new int[]{4,cols,rows});
		out[6] = temp12.tensorAlongDimension(0, 1,2);
		out[7] = temp12.tensorAlongDimension(2, 1,2);

		INDArray temp20 = Nd4j.rand(new int[]{rows,4,cols});
		out[8] = temp20.tensorAlongDimension(0, 2,0);
		out[9] = temp20.tensorAlongDimension(2, 2,0);

		INDArray temp21 = Nd4j.rand(new int[]{4,rows,cols});
		out[10] = temp21.tensorAlongDimension(0, 2,1);
		out[11] = temp21.tensorAlongDimension(2, 2,1);

		return out;
	}

	public static INDArray getPermutedWithShape(int rows, int cols){
		INDArray arr = Nd4j.rand(new int[]{cols,rows});
		return arr.permute(1,0);
	}

	public static INDArray getReshapedWithShape(int rows, int cols){
		int[] origShape = new int[3];
		if(rows%2 == 0){
			origShape[0] = rows/2;
			origShape[1] = cols;
			origShape[2] = 2;
		} else if( cols%2 == 0 ){
			origShape[0] = rows;
			origShape[1] = cols/2;
			origShape[2] = 2;
		} else {
			origShape[0] = 1;
			origShape[1] = rows;
			origShape[2] = cols;
		}
		INDArray orig = Nd4j.rand(origShape);
		return orig.reshape(rows,cols);
	}
}
