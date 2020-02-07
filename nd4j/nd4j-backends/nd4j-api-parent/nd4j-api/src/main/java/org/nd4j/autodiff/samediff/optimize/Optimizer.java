package org.nd4j.autodiff.samediff.optimize;

import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;

import java.util.Properties;

/**
 *
 * @author Alex Black
 */
public interface Optimizer {

    /**
     *
     * @param sd Current SameDiff instance to optimize
     * @param optimizationConfig Optimization configuration
     * @param op Operation to check for optimization
     * @param constantArrays
     * @param variablesArrays
     * @return True if the optimization was applied
     */
    boolean checkAndApply(SameDiff sd, Properties optimizationConfig, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays);

}
