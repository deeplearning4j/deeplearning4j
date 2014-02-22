package org.deeplearning4j.util;

import org.apache.commons.math3.distribution.UniformRealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.deeplearning4j.datasets.DataSet;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;
import org.jblas.SimpleBlas;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class MatrixUtil {
	private static Logger log = LoggerFactory.getLogger(MatrixUtil.class);
	
	
	
	public static void complainAboutMissMatchedMatrices(DoubleMatrix d1,DoubleMatrix d2) {
		if(d1 == null || d2 == null)
			throw new IllegalArgumentException("No null matrices allowed");
		if(d1.rows != d2.rows)
			throw new IllegalArgumentException("Matrices must have same rows");
		
	}
	
	
	public static DataSet xorData(int n) {
		
		DoubleMatrix x = DoubleMatrix.rand(n,2);
		x = x.gti(0.5);
		
		DoubleMatrix y = DoubleMatrix.zeros(n,2);
		for(int i = 0; i < x.rows; i++) {
			if(x.get(i,0) == x.get(i,1))
				y.put(i,0,1);
			else 
				y.put(i,1,1);
		}
		
		return new DataSet(x,y);
	
	}
	
	public static DataSet xorData(int n, int columns) {
		
		DoubleMatrix x = DoubleMatrix.rand(n,columns);
		x = x.gti(0.5);
		
		DoubleMatrix x2 = DoubleMatrix.rand(n,columns);
		x2 = x2.gti(0.5);
		
		DoubleMatrix eq = x.eq(x2).eq(DoubleMatrix.zeros(n,columns));
	
		
		int median = columns / 2;
		
		DoubleMatrix outcomes = new DoubleMatrix(n,2);
		for(int i = 0; i < outcomes.rows; i++) {
			DoubleMatrix left = eq.get(i,new org.jblas.ranges.IntervalRange(0,median));
			DoubleMatrix right = eq.get(i,new org.jblas.ranges.IntervalRange(median,columns));
			if(left.sum() > right.sum())
				outcomes.put(i,0,1);
			else
				outcomes.put(i,1,1);
		}
		
		
		return new DataSet(eq,outcomes);
	
	}
	
	public static double magnitude(DoubleMatrix vec) { 
		double sum_mag = 0; 
		for(int i = 0; i < vec.length;i++) 
			sum_mag = sum_mag + vec.get(i) * vec.get(i); 

		return Math.sqrt(sum_mag); 
	} 

	
	public static DoubleMatrix unroll(DoubleMatrix d) {
		DoubleMatrix ret = new DoubleMatrix(1,d.length);
		for(int i = 0; i < d.length; i++)
			ret.put(i,d.get(i));
		return ret;
	}
	
	
	public static DoubleMatrix outcomes(DoubleMatrix d) {
		DoubleMatrix ret = new DoubleMatrix(d.rows,1);
		for(int i = 0; i < d.rows; i++)
			ret.put(i,SimpleBlas.iamax(d.getRow(i)));
		return ret;
	}
	
	public static double cosineSim(DoubleMatrix d1,DoubleMatrix d2) {
		d1 = MatrixUtil.unitVec(d1);
		d2 = MatrixUtil.unitVec(d2);
		double ret = d1.dot(d2);
		return ret;
	}
	
	
	public static DoubleMatrix normalize(DoubleMatrix input) {
		double min = input.min();
		double max = input.max();
		return input.subi(min).divi(max - min);
	}
	
	
	public static double cosine(DoubleMatrix matrix) {
		//1.0 * math.sqrt(sum(val * val for val in vec1.itervalues()))
		return 1 * Math.sqrt(MatrixFunctions.pow(matrix, 2).sum());
	}

	
	public static DoubleMatrix unitVec(DoubleMatrix toScale) {
		double length = toScale.norm2();
		if(length > 0)
			return SimpleBlas.scal(1.0 / length, toScale);
		return toScale;
	}
	
	public static DoubleMatrix uniform(RandomGenerator rng,int rows,int columns) {

		UniformRealDistribution uDist = new UniformRealDistribution(rng,0,1);
		DoubleMatrix U = new DoubleMatrix(rows,columns);
		for(int i = 0; i < U.rows; i++)
			for(int j = 0; j < U.columns; j++) 
				U.put(i,j,uDist.sample());
		return U;
	}
	
	public static boolean isValidOutcome(DoubleMatrix out) {
		boolean found = false;
		for(int col = 0; col < out.length; col++) {
		     if(out.get(col) > 0) {
		    	 found = true;
		    	 break;
		     }
		}
		return found;
	}
	
	
	public static void ensureValidOutcomeMatrix(DoubleMatrix out) {
		boolean found = false;
		for(int col = 0; col < out.length; col++) {
		     if(out.get(col) > 0) {
		    	 found = true;
		    	 break;
		     }
		}
		if(!found) {
			log.warn("Found invalid matrix assuming; nothing which means adding a 1 to the first spot");
			out.put(0,1.0);
		}
		
	}
	
	public static void assertIntMatrix(DoubleMatrix matrix) {
		for(int i = 0; i < matrix.length; i++) {
			int cast = (int) matrix.get(i);
			if(cast != matrix.get(i))
				throw new IllegalArgumentException("Found something that is not an integer at linear index " + i);
		}
	}

	public static boolean isInfinite(DoubleMatrix test) {
		DoubleMatrix nan = test.isInfinite();
		for(int i = 0; i < nan.length; i++) {
			if(nan.get(i) > 0)
				return true;
		}
		return false;
	}

	public static boolean isNaN(DoubleMatrix test) {
		for(int i = 0; i < test.length; i++) {
			if(Double.isNaN(test.get(i)))
				return true;
		}
		return false;
	}




	public static void discretizeColumns(DoubleMatrix toDiscretize,int numBins) {
		DoubleMatrix columnMaxes = toDiscretize.columnMaxs();
		DoubleMatrix columnMins = toDiscretize.columnMins();
		for(int i = 0; i < toDiscretize.columns; i++) {
			double min = columnMins.get(i);
			double max = columnMaxes.get(i);
			DoubleMatrix col = toDiscretize.getColumn(i);
			DoubleMatrix newCol = new DoubleMatrix(col.length);
			for(int j = 0; j < col.length; j++) {
				int bin = MathUtils.discretize(col.get(j), min, max, numBins);
				newCol.put(j,bin);
			}
			toDiscretize.putColumn(i,newCol);

		}
	}


	public static DoubleMatrix roundToTheNearest(DoubleMatrix d,double num) {
		DoubleMatrix ret = d.mul(num);
		for(int i = 0; i < d.rows; i++)
			for(int j = 0; j < d.columns; j++) {
				double newNum = Math.round(d.get(i,j) * num);
				newNum /= num;
				ret.put(i,j,newNum);
			}
		return ret;
	}


	public static void columnNormalizeBySum(DoubleMatrix x) {
		for(int i = 0; i < x.columns; i++)
			x.putColumn(i, x.getColumn(i).div(x.getColumn(i).sum()));
	}


	public static DoubleMatrix toOutcomeVector(int index,int numOutcomes) {
		int[] nums = new int[numOutcomes];
		nums[index] = 1;
		return toMatrix(nums);
	}

	public static DoubleMatrix toMatrix(int[][] arr) {
		DoubleMatrix d = new DoubleMatrix(arr.length,arr[0].length);
		for(int i = 0; i < arr.length; i++)
			for(int j = 0; j < arr[i].length; j++)
				d.put(i,j,arr[i][j]);
		return d;
	}

	public static DoubleMatrix toMatrix(int[] arr) {
		DoubleMatrix d = new DoubleMatrix(arr.length);
		for(int i = 0; i < arr.length; i++)
			d.put(i,arr[i]);
		d.reshape(1, d.length);
		return d;
	}

	public static DoubleMatrix add(DoubleMatrix a,DoubleMatrix b) {
		return a.addi(b);
	}

	public static DoubleMatrix softmax(DoubleMatrix input) {
		DoubleMatrix max = input.rowMaxs();
        MatrixFunctions.expi(input.subiColumnVector(max));
        DoubleMatrix sum = input.rowSums();
        return input.diviColumnVector(sum);
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
		DoubleMatrix ones = DoubleMatrix.ones(x.rows, x.columns);
		return ones.div(ones.add(MatrixFunctions.exp(x.neg())));
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

	/**
	 * Calculates the column wise standard deviations
	 * of the matrix
	 * @param m the matrix to use
	 * @return the standard deviations of each column in the matrix
	 * as a row matrix
	 */
	public static DoubleMatrix columnStd(DoubleMatrix m) {
		DoubleMatrix ret = new DoubleMatrix(1,m.columns);
		for(int i = 0; i < m.columns; i++) {
			StandardDeviation std = new StandardDeviation();
			double result = std.evaluate(m.getColumn(i).data);
			ret.put(i,result);
		}
		return ret;
	}

	/**
	 * Returns the mean squared error of the 2 matrices.
	 * Note that the matrices must be the same length
	 * or an {@link IllegalArgumentException} is thrown
	 * @param input the first one
	 * @param other the second one
	 * @return the mean square error of the matrices
	 */
	public static double meanSquaredError(DoubleMatrix input,DoubleMatrix other) {
		if(input.length != other.length)
			throw new IllegalArgumentException("Matrices must be same length");
		SimpleRegression r = new SimpleRegression();
		r.addData(new double[][]{input.data,other.data});
		return r.getMeanSquareError();
	}

	public static DoubleMatrix log(DoubleMatrix vals) {
		DoubleMatrix ret = new DoubleMatrix(vals.rows,vals.columns);
		for(int i = 0; i < vals.length; i++) {
			ret.put(i,vals.get(i) == 0 ? 0 : Math.log(vals.get(i)));
		}
		return ret;
	}

	/**
	 * Returns the sum squared error of the 2 matrices.
	 * Note that the matrices must be the same length
	 * or an {@link IllegalArgumentException} is thrown
	 * @param input the first one
	 * @param other the second one
	 * @return the sum square error of the matrices
	 */
	public static double sumSquaredError(DoubleMatrix input,DoubleMatrix other) {
		if(input.length != other.length)
			throw new IllegalArgumentException("Matrices must be same length");
		SimpleRegression r = new SimpleRegression();
		r.addData(new double[][]{input.data,other.data});
		return r.getSumSquaredErrors();
	}

	public static void normalizeMatrix(DoubleMatrix toNormalize) {
		DoubleMatrix columnMeans = toNormalize.columnMeans();
		toNormalize.subiRowVector(columnMeans);
		toNormalize.diviRowVector(columnStd(toNormalize));
	}
	
	
	public static DoubleMatrix normalizeByColumnSums(DoubleMatrix m) {
		DoubleMatrix columnSums = m.columnSums();
		for(int i = 0; i < m.columns; i++) {
			m.putColumn(i,m.getColumn(i).div(columnSums.get(i)));

		}
		return m;
	}
	
	
	public static DoubleMatrix columnStdDeviation(DoubleMatrix m) {
		DoubleMatrix ret = new DoubleMatrix(1,m.columns);
		
		for(int i = 0; i < ret.length; i++) {
			StandardDeviation dev = new StandardDeviation();
			double std = dev.evaluate(m.getColumn(i).toArray());
			if(Double.isNaN(std))
				log.warn("WTF");
			ret.put(i,std);
		}
		
		return ret;
	}
	
	/**
	 * Divides the given matrix's columns
	 * by each column's respective standard deviations
	 * @param m the matrix to divide
	 * @return the column divided by the standard deviation
	 */
	public static DoubleMatrix divColumnsByStDeviation(DoubleMatrix m) {
		DoubleMatrix std = columnStdDeviation(m);
		for(int i = 0; i < m.columns; i++) {
			m.putColumn(i,m.getColumn(i).div(std.get(i)));
		}
		return m;
		
	}
	
	/**
	 * Subtracts by column mean.
	 * This ensures a mean of zero.
	 * This is part of normalizing inputs
	 * for a neural net
	 * @param m the matrix to normalize
	 * @return the normalized matrix which each
	 * column subtracted by its mean
	 */
	public static DoubleMatrix normalizeByColumnMeans(DoubleMatrix m) {
		DoubleMatrix columnMeans = m.columnMeans();
		for(int i = 0; i < m.columns; i++) {
			m.putColumn(i,m.getColumn(i).sub(columnMeans.get(i)));

		}
		return m;
	}
	
	public static DoubleMatrix normalizeByRowSums(DoubleMatrix m) {
		DoubleMatrix rowSums = m.rowSums();
		for(int i = 0; i < m.rows; i++) {
			m.putRow(i,m.getRow(i).div(rowSums.get(i)));
		}
		return m;
	}

}
