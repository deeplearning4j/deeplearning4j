package org.deeplearning4j.nn;


import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.activation.RectifiedLinear;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.util.ArrayUtil;


/**
 * Weight initialization utility
 * @author Adam Gibson
 */
public class WeightInitUtil {




    public static INDArray initWeights(int[] shape,float min,float max) {
         return Nd4j.rand(shape,min,max,new MersenneTwister(123));
    }


    /**
     * Initializes a matrix with the given weight initialization scheme
     * @param nIn the number of rows in the matrix
     * @param nOut the number of columns in the matrix
     * @param initScheme the scheme to use
     * @return a matrix of the specified dimensions with the specified
     * distribution based on the initialization scheme
     */
    public static INDArray initWeights(int nIn,int nOut,WeightInit initScheme,ActivationFunction act,RealDistribution dist) {
        INDArray ret = Nd4j.randn(nIn,nOut);
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
                    INDArray perm = Nd4j.create(ArrayUtil.randomPermutation(nIn)).reshape(new int[]{nIn});
                    INDArray subIndices = perm.get(NDArrayIndex.interval(fanOut, perm.length()), NDArrayIndex.interval(0, nOut));
                    INDArray indices = perm.get(new NDArrayIndex(ArrayUtil.toInts(subIndices)),new NDArrayIndex(new int[]{i}));
                    INDArray randInit = ret.get(ArrayUtil.toInts(perm.get(ArrayUtil.toInts(indices)).muli(Nd4j.scalar(smallVal))));
                    if(randInit.length() != 0)
                        ret.put(ArrayUtil.toInts(indices),randInit);

                }

                ret.muli(Nd4j.scalar(initCoeff)).addi(Nd4j.scalar(shift));
                return ret;



            case  VI:
                double r = Math.sqrt(6) / Math.sqrt(nIn + nOut + 1);
                ret.muli(Nd4j.scalar(2)).muli(Nd4j.scalar(r)).subi(Nd4j.scalar(r));
                return ret;

            case DISTRIBUTION:
                for(int i = 0; i < ret.rows(); i++) {
                    ret.putRow(i,Nd4j.create(dist.sample(ret.columns())));
                }
                return ret;



        }

        throw new IllegalStateException("Illegal weight init value");
    }


}
