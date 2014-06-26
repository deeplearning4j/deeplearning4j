package org.deeplearning4j.nn;

import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.nn.activation.RectifiedLinear;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.RangeUtils;
import org.jblas.util.Permutations;

/**
 * Weight initialization utility
 * @author Adam Gibson
 */
public class WeightInitUtil {


    /**
     * Initializes a matrix with the given weight initialization scheme
     * @param nIn the number of rows in the matrix
     * @param nOut the number of columns in the matrix
     * @param initScheme the scheme to use
     * @param maxNonZeroPerColumn useful for only si
     * @return a matrix of the specified dimensions with the specified
     * distribution based on the initialization scheme
     */
    public static DoubleMatrix initWeights(int nIn,int nOut,WeightInit initScheme,ActivationFunction act, int maxNonZeroPerColumn) {
        DoubleMatrix ret = DoubleMatrix.randn(nIn,nOut);
        switch(initScheme) {
            case SI:
                double shift = 0,initCoeff = 1;
                int smallVal = 0;
                if(act instanceof RectifiedLinear) {
                    shift = 1e-1;
                    initCoeff = 0.25;
                }
                double fanOut = nOut;
                for(int i = 0; i < nOut; i++) {
                    int[] perm = Permutations.randomPermutation(nIn);
                    for(int j = 0; j < perm.length; j++) {
                        ret.put(perm[j],j,ret.get(perm[j],j) * smallVal);
                    }

                }

                ret.muli(initCoeff).add(shift);
                return ret;



            case  VI:
                double r = Math.sqrt(6) / Math.sqrt(nIn + nOut + 1);
                ret.muli(2).muli(r).subi(r);
                return ret;



        }

        throw new IllegalStateException("Illegal weight init value");
    }




    /**
     * Initializes a matrix with the given weight initialization scheme
     * @param nIn the number of rows in the matrix
     * @param nOut the number of columns in the matrix
     * @param initScheme the scheme to use
     * @return a matrix of the specified dimensions with the specified
     * distribution based on the initialization scheme
     */
    public static DoubleMatrix initWeights(int nIn,int nOut,WeightInit initScheme,ActivationFunction act) {
        DoubleMatrix ret = DoubleMatrix.randn(nIn,nOut);
        switch(initScheme) {
            case SI:
                int maxNonZeroPerColumn = 15;
                double shift = 0,initCoeff = 1;
                int smallVal = 0;
                if(act instanceof RectifiedLinear) {
                    shift = 1e-1;
                    initCoeff = 0.25;
                }

                int fanOut = Math.min(maxNonZeroPerColumn,nOut);
                for(int i = 0; i < nOut; i++) {
                    DoubleMatrix perm = MatrixUtil.toMatrix(Permutations.randomPermutation(nIn)).reshape(nIn,1);
                    DoubleMatrix subIndices = perm.get(RangeUtils.interval(fanOut,perm.length),RangeUtils.all());
                    DoubleMatrix indices = perm.get(perm.get(subIndices,MatrixUtil.toMatrix(new int[]{i})));
                    DoubleMatrix randInit = ret.get(perm.get(indices).muli(smallVal));
                    if(randInit.length != 0)
                        ret.put(indices,randInit);

                }

                ret.muli(initCoeff).add(shift);
                return ret;



            case  VI:
                double r = Math.sqrt(6) / Math.sqrt(nIn + nOut + 1);
                ret.muli(2).muli(r).subi(r);
                return ret;



        }

        throw new IllegalStateException("Illegal weight init value");
    }


}
