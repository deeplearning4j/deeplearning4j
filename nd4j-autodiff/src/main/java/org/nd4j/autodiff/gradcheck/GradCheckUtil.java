package org.nd4j.autodiff.gradcheck;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.tensorgrad.TensorGrad;
import org.nd4j.autodiff.tensorgrad.impl.TensorGradFunction;
import org.nd4j.autodiff.tensorgrad.impl.TensorGradVariable;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by agibsonccc on 5/1/17.
 */
@Slf4j
public class GradCheckUtil {


    /**
     *
     * @param wrt
     * @param epsilon
     * @param maxRelError
     * @param minAbsoluteError
     * @param print
     * @param exitOnFirstError
     * @param input
     * @param rngSeed
     * @return
     */
    public static boolean checkGradients(TensorGradVariable wrt,
                                         double epsilon,
                                         double maxRelError,
                                         double minAbsoluteError,
                                         boolean print,
                                         boolean exitOnFirstError,
                                         TensorGradVariable input,
                                         int rngSeed) {
        //Basic sanity checks on input:
        if (epsilon <= 0.0 || epsilon > 0.1)
            throw new IllegalArgumentException("Invalid epsilon: expect epsilon in range (0,0.1], usually 1e-4 or so");
        if (maxRelError <= 0.0 || maxRelError > 0.25)
            throw new IllegalArgumentException("Invalid maxRelativeError: " + maxRelError);

        DataBuffer.Type dataType = DataTypeUtil.getDtypeFromContext();
        if (dataType != DataBuffer.Type.DOUBLE) {
            throw new IllegalStateException("Cannot perform gradient check: Datatype is not set to double precision ("
                    + "is: " + dataType + "). Double precision must be used for gradient checks. Set "
                    + "DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE); before using GradientCheckUtil");
        }

        TensorGradVariable grad = input.getTensorGrad().grad(input,wrt);
        TensorGradVariable gradientToCheck = grad.dup();
        TensorGradVariable originalParams = input.dup();

        int nParams = ArrayUtil.prod(originalParams.getShape());

        int totalNFailures = 0;
        double maxError = 0.0;
 /*       *//**
         * Need to figure out how to avoid the graph changing
         *
         *//*
        for (int i = 0; i < nParams; i++) {
              //(w+epsilon): Do forward pass and score
            double origValue = params.getDouble(i);
            params.putScalar(i, origValue + epsilon);

            //TODO add a 'score' method that doesn't calculate gradients...
            Nd4j.getRandom().setSeed(rngSeed);
            layer.computeGradientAndScore();
            double scorePlus = layer.score();

            //(w-epsilon): Do forward pass and score
            params.putScalar(i, origValue - epsilon);
            Nd4j.getRandom().setSeed(rngSeed);
            layer.computeGradientAndScore();
            double scoreMinus = layer.score();

            //Reset original param value
            params.putScalar(i, origValue);

            //Calculate numerical parameter gradient:
            double scoreDelta = scorePlus - scoreMinus;

            double numericalGradient = scoreDelta / (2 * epsilon);
            if (Double.isNaN(numericalGradient))
                throw new IllegalStateException("Numerical gradient was NaN for parameter " + i + " of " + nParams);

            double backpropGradient = gradientToCheck.getDouble(i);
            //http://cs231n.github.io/neural-networks-3/#gradcheck
            //use mean centered
            double relError = Math.abs(backpropGradient - numericalGradient)
                    / (Math.abs(numericalGradient) + Math.abs(backpropGradient));
            if (backpropGradient == 0.0 && numericalGradient == 0.0)
                relError = 0.0; //Edge case: i.e., RNNs with time series length of 1.0

            if (relError > maxError)
                maxError = relError;
            if (relError > maxRelError || Double.isNaN(relError)) {
                double absError = Math.abs(backpropGradient - numericalGradient);
                if (absError < minAbsoluteError) {
                    log.info("Param " + i +  " passed: grad= " + backpropGradient
                            + ", numericalGrad= " + numericalGradient + ", relError= " + relError
                            + "; absolute error = " + absError + " < minAbsoluteError = " + minAbsoluteError);
                } else {
                    if (print)
                        log.info("Param " + i + " FAILED: grad= " + backpropGradient
                                + ", numericalGrad= " + numericalGradient + ", relError= " + relError
                                + ", scorePlus=" + scorePlus + ", scoreMinus= " + scoreMinus);
                    if (exitOnFirstError)
                        return false;
                    totalNFailures++;
                }
            } else if (print) {
                log.info("Param " + i + "passed: grad= " + backpropGradient + ", numericalGrad= "
                        + numericalGradient + ", relError= " + relError);
            }
        }

        if (print) {
            int nPass = nParams - totalNFailures;
            log.info("GradientCheckUtil.checkGradients(): " + nParams + " params checked, " + nPass + " passed, "
                    + totalNFailures + " failed. Largest relative error = " + maxError);
        }
*/
        return totalNFailures == 0;
    }
}
