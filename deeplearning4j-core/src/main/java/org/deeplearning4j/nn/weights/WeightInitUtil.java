package org.deeplearning4j.nn.weights;


import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.deeplearning4j.distributions.Distributions;
import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/**
 * Weight initialization utility
 * @author Adam Gibson
 */
public class WeightInitUtil {


    /**
     * Normalized weight init
     * @param shape shape
     * @param nIn number of inputs
     * @return the weights
     */
    public static INDArray normalized(int[] shape,int nIn) {
        return Nd4j.rand(shape).subi(0.5).divi((double) nIn);
    }

    /**
     * Generate a random matrix with respect to the number of inputs and outputs.
     * This is a bound uniform distribution with the specified minimum and maximum
     * @param shape the shape of the matrix
     * @param nIn the number of inputs
     * @param nOut the number of outputs
     * @return
     */
    public static INDArray uniformBasedOnInAndOut(int[] shape,int nIn,int nOut) {
        double min = -4.0 * Math.sqrt(6.0 / (double) (nOut + nIn));
        double max = 4.0 * Math.sqrt(6.0 / (double) (nOut + nIn));
        return Nd4j.rand(shape, Distributions.uniform(new MersenneTwister(123),min,max));
    }

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
        INDArray ret = null;
        switch(initScheme) {
            case NORMALIZED:
                ret = Nd4j.rand(nIn,nOut);
                return ret.subi(0.5).divi(nIn);
            case UNIFORM:
                double a = 1 / nIn;
                return Nd4j.rand(new int[]{nIn,nOut},-a,a,new MersenneTwister(123));

            case  VI:
                ret = Nd4j.rand(nIn,nOut);
                double r = Math.sqrt(6) / Math.sqrt(nIn + nOut + 1);
                ret.muli(2).muli(r).subi(r);
                return ret;

            case DISTRIBUTION:
                ret = Nd4j.rand(nIn,nOut);
                for(int i = 0; i < ret.rows(); i++) {
                    ret.putRow(i,Nd4j.create(dist.sample(ret.columns())));
                }
                return ret;
            case SIZE:
                return uniformBasedOnInAndOut(new int[]{nIn,nOut},nIn,nOut);
            case ZERO:
                return Nd4j.create(new int[]{nIn,nOut});



        }

        throw new IllegalStateException("Illegal weight init value");
    }


}
