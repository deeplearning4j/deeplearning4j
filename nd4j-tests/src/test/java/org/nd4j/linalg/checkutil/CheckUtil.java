package org.nd4j.linalg.checkutil;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
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
	public static boolean checkMmul(INDArray first, INDArray second, double maxRelativeDifference, double minAbsDifference) {
		if(first.size(1) != second.size(0)) throw new IllegalArgumentException("first.columns != second.rows");
		RealMatrix rmFirst = convertToApacheMatrix(first);
		RealMatrix rmSecond = convertToApacheMatrix(second);

		INDArray result = first.mmul(second);
		RealMatrix rmResult = rmFirst.multiply(rmSecond);

		if(!checkShape(rmResult,result)) return false;
        boolean ok = checkEntries(rmResult, result, maxRelativeDifference, minAbsDifference);
        if(!ok) {
            INDArray onCopies = Shape.toOffsetZeroCopy(first).mmul(Shape.toOffsetZeroCopy(second));
            printFailureDetails(first, second, rmResult, result, onCopies, "mmul");
        }
        return ok;
	}

	public static boolean checkGemm(INDArray a, INDArray b, INDArray c, boolean transposeA, boolean transposeB,
									double alpha, double beta,
									double maxRelativeDifference, double minAbsDifference) {
		int commonDimA = (transposeA ? a.rows() : a.columns());
		int commonDimB = (transposeB ? b.columns() : b.rows());
		if(commonDimA != commonDimB)
            throw new IllegalArgumentException("Common dimensions don't match: a.shape=" +
                Arrays.toString(a.shape()) + ", b.shape="+ Arrays.toString(b.shape()) + ", tA=" + transposeA + ", tb=" + transposeB);
		int outRows = (transposeA ? a.columns() : a.rows());
		int outCols = (transposeB ? b.rows() : b.columns());
		if(c.rows() != outRows || c.columns() != outCols)
            throw new IllegalArgumentException("C does not match outRows or outCols");
		if(c.offset() != 0 || c.ordering() != 'f')
            throw new IllegalArgumentException("Invalid c");

        INDArray aConvert = transposeA ? a.transpose() : a;
        RealMatrix rmA = convertToApacheMatrix(aConvert);
        INDArray bConvet = transposeB ? b.transpose() : b;
		RealMatrix rmB = convertToApacheMatrix(bConvet);
		RealMatrix rmC = convertToApacheMatrix(c);
		RealMatrix rmExpected = rmA.scalarMultiply(alpha).multiply(rmB).add(rmC.scalarMultiply(beta));
        INDArray cCopy1 = Nd4j.create(c.shape(), 'f');
        cCopy1.assign(c);
        INDArray cCopy2 = Nd4j.create(c.shape(), 'f');
        cCopy2.assign(c);

		INDArray out = Nd4j.gemm(a, b, c, transposeA, transposeB, alpha, beta);
        if(out != c) {
            System.out.println("Returned different array than c");
            return false;
        }
		if(!checkShape(rmExpected,out))
            return false;
		boolean ok = checkEntries(rmExpected,out,maxRelativeDifference,minAbsDifference);
		if(!ok) {
			INDArray aCopy = Shape.toOffsetZeroCopy(a);
			INDArray bCopy = Shape.toOffsetZeroCopy(b);
			INDArray onCopies = Nd4j.gemm(aCopy, bCopy, cCopy1, transposeA, transposeB, alpha, beta);
			printGemmFailureDetails(a,b,cCopy2,transposeA,transposeB,alpha,beta,rmExpected,out,onCopies);
		}
		return ok;
	}

	/**Same as checkMmul, but for matrix addition */
	public static boolean checkAdd(INDArray first, INDArray second, double maxRelativeDifference, double minAbsDifference) {
		RealMatrix rmFirst = convertToApacheMatrix(first);
        RealMatrix rmSecond = convertToApacheMatrix(second);

		INDArray result = first.add(second);
		RealMatrix rmResult = rmFirst.add(rmSecond);

        if (!checkShape(rmResult, result)) return false;
        boolean ok = checkEntries(rmResult,result,maxRelativeDifference,minAbsDifference);
        if(!ok){
            INDArray onCopies = Shape.toOffsetZeroCopy(first).add(Shape.toOffsetZeroCopy(second));
            printFailureDetails(first, second, rmResult, result, onCopies, "add");
        }
        return ok;
	}

	/** Same as checkMmul, but for matrix subtraction */
	public static boolean checkSubtract(INDArray first, INDArray second, double maxRelativeDifference, double minAbsDifference ){
		RealMatrix rmFirst = convertToApacheMatrix(first);
		RealMatrix rmSecond = convertToApacheMatrix(second);

		INDArray result = first.sub(second);
		RealMatrix rmResult = rmFirst.subtract(rmSecond);

		if(!checkShape(rmResult,result)) return false;
		boolean ok = checkEntries(rmResult, result, maxRelativeDifference, minAbsDifference);
        if(!ok){
            INDArray onCopies = Shape.toOffsetZeroCopy(first).sub(Shape.toOffsetZeroCopy(second));
            printFailureDetails(first, second, rmResult, result, onCopies, "sub");
        }
        return ok;
	}

	public static boolean checkMulManually(INDArray first, INDArray second, double maxRelativeDifference, double minAbsDifference ){
		//No apache commons element-wise multiply, but can do this manually

		INDArray result = first.mul(second);
		int[] shape = first.shape();

		INDArray expected = Nd4j.zeros(first.shape());

		for(int i=0; i<shape[0]; i++ ) {
            for (int j = 0; j < shape[1]; j++) {
                double v = first.getDouble(i, j) * second.getDouble(i, j);
                expected.putScalar(new int[]{i, j}, v);
            }
        }
        if(!checkShape(expected,result)) return false;
        boolean ok = checkEntries(expected, result, maxRelativeDifference, minAbsDifference);
        if(!ok){
            INDArray onCopies = Shape.toOffsetZeroCopy(first).mul(Shape.toOffsetZeroCopy(second));
            printFailureDetails(first,second,expected,result,onCopies,"mul");
        }
        return ok;
	}

    public static boolean checkDivManually(INDArray first, INDArray second, double maxRelativeDifference, double minAbsDifference ){
        //No apache commons element-wise division, but can do this manually

        INDArray result = first.div(second);
        int[] shape = first.shape();

        INDArray expected = Nd4j.zeros(first.shape());

        for(int i=0; i<shape[0]; i++ ) {
            for (int j = 0; j < shape[1]; j++) {
                double v = first.getDouble(i, j) / second.getDouble(i, j);
                expected.putScalar(new int[]{i, j}, v);
            }
        }
        if(!checkShape(expected,result)) return false;
        boolean ok = checkEntries(expected, result, maxRelativeDifference, minAbsDifference);
        if(!ok){
            INDArray onCopies = Shape.toOffsetZeroCopy(first).mul(Shape.toOffsetZeroCopy(second));
            printFailureDetails(first, second, expected, result, onCopies, "div");
        }
        return ok;
    }

	private static boolean checkShape(RealMatrix rmResult, INDArray result){
		int[] outShape = {rmResult.getRowDimension(),rmResult.getColumnDimension()};
		if(!Arrays.equals(outShape, result.shape())){
			System.out.println("Failure on shape: " + Arrays.toString(result.shape()) + ", expected " + Arrays.toString(outShape));
			return false;
		}
		return true;
	}

    private static boolean checkShape(INDArray expected, INDArray actual){
        if(!Arrays.equals(expected.shape(), actual.shape())){
            System.out.println("Failure on shape: " + Arrays.toString(actual.shape()) + ", expected " + Arrays.toString(expected.shape()));
            return false;
        }
        return true;
    }

	public static boolean checkEntries(RealMatrix rmResult, INDArray result, double maxRelativeDifference, double minAbsDifference){
		int[] outShape = {rmResult.getRowDimension(),rmResult.getColumnDimension()};
		for( int i = 0; i < outShape[0]; i++) {
			for( int j = 0; j < outShape[1]; j++) {
				double expOut = rmResult.getEntry(i, j);
				double actOut = result.getDouble(i,j);
				if(expOut==0.0 && actOut==0.0)
                    continue;
				double absError = Math.abs(expOut - actOut);
				double relError = absError / (Math.abs(expOut) + Math.abs(actOut));
				if (relError > maxRelativeDifference && absError > minAbsDifference) {
                    System.out.println("Failure on value: (" +i+","+j+" exp="+expOut + ", act="+actOut + ", absError="+absError + ", relError="+relError);
					return false;
				}
			}
		}
		return true;
	}

    public static boolean checkEntries(INDArray expected, INDArray actual, double maxRelativeDifference, double minAbsDifference){
        int[] outShape = expected.shape();
        for( int i = 0; i < outShape[0]; i++) {
            for( int j = 0; j < outShape[1]; j++) {
                double expOut = expected.getDouble(i, j);
                double actOut = actual.getDouble(i,j);
                if(expOut==0.0 && actOut==0.0)
                    continue;
                double absError = Math.abs(expOut - actOut);
                double relError = absError / (Math.abs(expOut) + Math.abs(actOut));
                if (relError > maxRelativeDifference && absError > minAbsDifference) {
                    System.out.println("Failure on value: (" +i+","+j+" exp="+expOut + ", act="+actOut + ", absError="+absError + ", relError="+relError);
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

	public static void printFailureDetails(INDArray first, INDArray second, RealMatrix expected, INDArray actual, INDArray onCopies, String op){
        System.out.println("\nFactory: " + Nd4j.factory().getClass() + "\n");

		System.out.println("First:");
        printMatrixFullPrecision(first);
        System.out.println("\nSecond:");
        printMatrixFullPrecision(second);
        System.out.println("\nExpected (Apache Commons)");
        printApacheMatrix(expected);
        System.out.println("\nSame Nd4j op on copies: (Shape.toOffsetZeroCopy(first)." + op + "(Shape.toOffsetZeroCopy(second)))");
        printMatrixFullPrecision(onCopies);
        System.out.println("\nActual:");
        printMatrixFullPrecision(actual);
	}

	public static void printGemmFailureDetails(INDArray a, INDArray b, INDArray c, boolean transposeA, boolean transposeB,
											   double alpha, double beta, RealMatrix expected, INDArray actual, INDArray onCopies){
		System.out.println("\nFactory: " + Nd4j.factory().getClass() + "\n");
		System.out.println("Op: gemm(a,b,c,transposeA="+transposeA+",transposeB="+transposeB+",alpha="+alpha+",beta="+beta+")");

		System.out.println("a:");
		printMatrixFullPrecision(a);
		System.out.println("\nb:");
		printMatrixFullPrecision(b);
		System.out.println("\nc:");
		printMatrixFullPrecision(c);
		System.out.println("\nExpected (Apache Commons)");
		printApacheMatrix(expected);
		System.out.println("\nSame Nd4j op on zero offset copies: gemm(aCopy,bCopy,cCopy," + transposeA + "," + transposeB + "," + alpha + "," + beta + ")");
		printMatrixFullPrecision(onCopies);
		System.out.println("\nActual:");
		printMatrixFullPrecision(actual);
	}

	public static void printMatrixFullPrecision(INDArray matrix){
		boolean floatType = (matrix.data().dataType() == DataBuffer.Type.FLOAT);
        printNDArrayHeader(matrix);
		int[] shape = matrix.shape();
		for( int i=0; i<shape[0]; i++ ){
			for( int j=0; j<shape[1]; j++ ){
				if(floatType) System.out.print(matrix.getFloat(i,j));
                else System.out.print(matrix.getDouble(i,j));
				if(j != shape[1]-1) System.out.print(", ");
				else System.out.println();
			}
		}
	}

    public static void printNDArrayHeader(INDArray array){
        System.out.println(array.data().dataType() + " - order=" + array.ordering() + ", offset=" + array.offset() +
                ", shape=" + Arrays.toString(array.shape()) + ", stride=" + Arrays.toString(array.stride())
                + ", length=" + array.length() + ", data().length()=" + array.data().length());
    }

    public static void printFailureDetails(INDArray first, INDArray second, INDArray expected, INDArray actual, INDArray onCopies, String op){
        System.out.println("\nFactory: " + Nd4j.factory().getClass() + "\n");

        System.out.println("First:");
        printMatrixFullPrecision(first);
        System.out.println("\nSecond:");
        printMatrixFullPrecision(second);
        System.out.println("\nExpected");
        printMatrixFullPrecision(expected);
        System.out.println("\nSame Nd4j op on copies: (Shape.toOffsetZeroCopy(first)." + op + "(Shape.toOffsetZeroCopy(second)))");
        printMatrixFullPrecision(onCopies);
        System.out.println("\nActual:");
        printMatrixFullPrecision(actual);
    }

    public static void printApacheMatrix(RealMatrix matrix){
        int nRows = matrix.getRowDimension();
        int nCols = matrix.getColumnDimension();
        System.out.println("Apache Commons RealMatrix: Shape: [" + nRows + "," + nCols + "]");
        for( int i=0; i<nRows; i++ ){
            for( int j=0; j<nCols; j++ ){
                System.out.print(matrix.getEntry(i,j));
                if( j != nCols-1 ) System.out.print(", ");
                else System.out.println();
            }
        }
    }
}
