package org.nd4j.linalg.dimensionalityreduction;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.GaussianDistribution;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Created by huitseeker on 7/28/17.
 */
public class RandomProjection {

    private int components;
    private Random rng;
    private double eps;
    private boolean autoMode;

    public RandomProjection(double eps, Random rng){
        this.rng = rng;
        this.eps = eps;
        this.autoMode = true;
    }

    public RandomProjection(double eps){
        this(eps, Nd4j.getRandom());
    }

    public RandomProjection(int components, Random rng){
        this.rng = rng;
        this.components = components;
        this.autoMode = false;
    }

    public RandomProjection(int components){
        this(components, Nd4j.getRandom());
    }

    /**
     * Find a safe number of components to project this to, through
     * the Johnson-Lindenstrauss lemma
     * The minimum number n' of components to guarantee the eps-embedding is
     * given by:
     *
     * n' >= 4 log(n) / (eps² / 2 - eps³ / 3)
     *
     * see http://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf §2.1
     * @param n Number of samples. If an array is given, it will compute
     *        a safe number of components array-wise.
     * @param eps Maximum distortion rate as defined by the Johnson-Lindenstrauss lemma.
     *            Will compute array-wise if an array is given.
     * @return
     */
    public static List<Integer> johnsonLindenstraussMinDim(int[] n, double... eps){
        Boolean basicCheck = n == null || n.length == 0 || eps == null || eps.length == 0;
        if (basicCheck)
            throw new IllegalArgumentException("Johnson-Lindenstrauss dimension estimation requires > 0 components and at least a relative error");
        for (double epsilon: eps){
            if (epsilon <= 0 || epsilon >= 1) {
                throw new IllegalArgumentException("A relative error should be in ]0, 1[");
            }
        }
        List<Integer> res = new ArrayList(n.length * eps.length);
        for (double epsilon : eps){
            double denom = (Math.pow(epsilon, 2) / 2 - Math.pow(epsilon, 3) / 3);
            for (int components: n){
                res.add((int) (4 * Math.log(components) / denom));
            }
        }
        return res;
    }

    public static List<Long> johnsonLindenstraussMinDim(long[] n, double... eps){
        Boolean basicCheck = n == null || n.length == 0 || eps == null || eps.length == 0;
        if (basicCheck)
            throw new IllegalArgumentException("Johnson-Lindenstrauss dimension estimation requires > 0 components and at least a relative error");
        for (double epsilon: eps){
            if (epsilon <= 0 || epsilon >= 1) {
                throw new IllegalArgumentException("A relative error should be in ]0, 1[");
            }
        }
        List<Long> res = new ArrayList(n.length * eps.length);
        for (double epsilon : eps){
            double denom = (Math.pow(epsilon, 2) / 2 - Math.pow(epsilon, 3) / 3);
            for (long components: n){
                res.add((long) (4 * Math.log(components) / denom));
            }
        }
        return res;
    }

    public static List<Integer> johnsonLindenStraussMinDim(int n, double... eps){
        return johnsonLindenstraussMinDim(new int[]{n}, eps);
    }

    public static List<Long> johnsonLindenStraussMinDim(long n, double... eps){
        return johnsonLindenstraussMinDim(new long[]{n}, eps);
    }

    /**
     * Generate a dense Gaussian random matrix.
     *
     *   The n' components of the random matrix are drawn from
     *       N(0, 1.0 / n').
     *
     * @param shape
     * @param rng
     * @return
     */
    private INDArray gaussianRandomMatrix(long[] shape, Random rng){
        Nd4j.checkShapeValues(shape);
        INDArray res = Nd4j.create(shape);

        GaussianDistribution op1 = new GaussianDistribution(res, 0.0, 1.0 / Math.sqrt(shape[0]));
        Nd4j.getExecutioner().exec(op1, rng);
        return res;
    }

    private long[] projectionMatrixShape;
    private INDArray _projectionMatrix;

    private INDArray getProjectionMatrix(long[] shape, Random rng){
        if (! Arrays.equals(projectionMatrixShape, shape) || _projectionMatrix == null)
            _projectionMatrix = gaussianRandomMatrix(shape, rng);
        return _projectionMatrix;
    }

    /**
     *
     * Compute the target shape of the projection matrix
     * @param shape the shape of the data tensor
     * @param eps the relative error used in the Johnson-Lindenstrauss estimation
     * @param auto whether to use JL estimation for user specification
     * @param targetDimension the target size for the
     *
     */
    private static int[] targetShape(int[] shape, double eps, int targetDimension, boolean auto){
        int components = targetDimension;
        if (auto) components = johnsonLindenStraussMinDim(shape[0], eps).get(0);
        // JL or user spec edge cases
        if (auto && (components <= 0 || components > shape[1])){
            throw new ND4JIllegalStateException(String.format("Estimation led to a target dimension of %d, which is invalid", components));
        }
        return new int[]{ shape[1], components};
    }

    private static long[] targetShape(long[] shape, double eps, int targetDimension, boolean auto){
        long components = targetDimension;
        if (auto) components = johnsonLindenStraussMinDim(shape[0], eps).get(0);
        // JL or user spec edge cases
        if (auto && (components <= 0 || components > shape[1])){
            throw new ND4JIllegalStateException(String.format("Estimation led to a target dimension of %d, which is invalid", components));
        }
        return new long[]{ shape[1], components};
    }


    /**
     * Compute the target shape of a suitable projection matrix
     * @param X the Data tensor
     * @param eps the relative error used in the Johnson-Lindenstrauss estimation
     * @return the shape of the projection matrix to use
     */
    protected static long[] targetShape(INDArray X, double eps) {
        return targetShape(X.shape(), eps, -1, true);
    }

    /**
     * Compute the target shape of a suitable projection matrix
     * @param X the Data Tensor
     * @param targetDimension a desired dimension
     * @return the shape of the projection matrix to use
     */
    protected static long[] targetShape(INDArray X, int targetDimension) {
        return targetShape(X.shape(), -1, targetDimension, false);
    }


    /**
     * Create a copy random projection by using matrix product with a random matrix
     * @param data
     * @return the projected matrix
     */
    public INDArray project(INDArray data){
        long[] tShape = targetShape(data.shape(), eps, components, autoMode);
        return data.mmul(getProjectionMatrix(tShape, this.rng));
    }

    /**
     * Create a copy random projection by using matrix product with a random matrix
     *
     * @param data
     * @param result a placeholder result
     * @return
     */
    public INDArray project(INDArray data, INDArray result){
        long[] tShape = targetShape(data.shape(), eps, components, autoMode);
        return data.mmuli(getProjectionMatrix(tShape, this.rng), result);
    }

    /**
     * Create an in-place random projection by using in-place matrix product with a random matrix
     * @param data
     * @return the projected matrix
     */
    public INDArray projecti(INDArray data){
        long[] tShape = targetShape(data.shape(), eps, components, autoMode);
        return data.mmuli(getProjectionMatrix(tShape, this.rng));
    }

    /**
     * Create an in-place random projection by using in-place matrix product with a random matrix
     *
     * @param data
     * @param result a placeholder result
     * @return
     */
    public INDArray projecti(INDArray data, INDArray result){
        long[] tShape = targetShape(data.shape(), eps, components, autoMode);
        return data.mmuli(getProjectionMatrix(tShape, this.rng), result);
    }



}
