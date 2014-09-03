package org.deeplearning4j.nn;


import org.apache.commons.math3.distribution.RealDistribution;
import org.deeplearning4j.linalg.api.activation.ActivationFunction;
import org.deeplearning4j.linalg.api.activation.RectifiedLinear;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.indexing.NDArrayIndex;
import org.deeplearning4j.linalg.util.ArrayUtil;


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
     * @return a matrix of the specified dimensions with the specified
     * distribution based on the initialization scheme
     */
    public static INDArray initWeights(int nIn,int nOut,WeightInit initScheme,ActivationFunction act,RealDistribution dist) {
        INDArray ret = NDArrays.randn(nIn,nOut);
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
                    INDArray perm = NDArrays.create(ArrayUtil.randomPermutation(nIn)).reshape(new int[]{nIn,1});
                    INDArray subIndices = perm.get(NDArrayIndex.interval(fanOut, perm.length()), NDArrayIndex.interval(0, nOut));
                    INDArray indices = perm.get(new NDArrayIndex(ArrayUtil.toInts(subIndices)),new NDArrayIndex(new int[]{i}));
                    INDArray randInit = ret.get(ArrayUtil.toInts(perm.get(ArrayUtil.toInts(indices)).muli(NDArrays.scalar(smallVal))));
                    if(randInit.length() != 0)
                        ret.put(ArrayUtil.toInts(indices),randInit);

                }

                ret.muli(NDArrays.scalar(initCoeff)).addi(NDArrays.scalar(shift));
                return ret;



            case  VI:
                double r = Math.sqrt(6) / Math.sqrt(nIn + nOut + 1);
                ret.muli(NDArrays.scalar(2)).muli(NDArrays.scalar(r)).subi(NDArrays.scalar(r));
                return ret;

            case DISTRIBUTION:
                for(int i = 0; i < ret.rows(); i++) {
                    ret.putRow(i,NDArrays.create(dist.sample(ret.columns())));
                }
                return ret;



        }

        throw new IllegalStateException("Illegal weight init value");
    }


}
