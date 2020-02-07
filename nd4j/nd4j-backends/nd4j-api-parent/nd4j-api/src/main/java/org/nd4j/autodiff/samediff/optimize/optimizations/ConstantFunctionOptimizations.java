package org.nd4j.autodiff.samediff.optimize.optimizations;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.internal.Variable;
import org.nd4j.autodiff.samediff.optimize.Optimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;

import java.util.List;
import java.util.Properties;

/**
 * This set of optimizations looks for functions that are applied to constants, and "pre executes" them, so they don't have
 * to be calculated (returning the same value) on each run.
 *
 * @author Alex Black
 */
public class ConstantFunctionOptimizations extends BaseOptimizerSet {

    public static final String CONSTANT_FN_FOLDING_MAX_SIZE = "optimizer.constants.function.max.output.size";
    public static final long CONSTANT_FN_FOLDING_MAX_SIZE_DEFAULT = 4 * 1024 * 1024;    //4MB

    public static class FoldConstantFunctions implements Optimizer {
        @Override
        public boolean checkAndApply(SameDiff sd, Properties optimizationConfig, SameDiffOp op, ArrayHolder constantArrays, ArrayHolder variablesArrays) {
            //TODO This function needs to check for non-deterministic ops - i.e., random ops - and not apply the optimization to these

            List<String> in = op.getInputsToOp();
            if (in == null || in.isEmpty())
                return false;
            for (String s : in) {
                if (!sd.getVariable(s).isConstant())
                    return false;
            }

            long maxSizeToApply = Long.parseLong(optimizationConfig.getProperty(CONSTANT_FN_FOLDING_MAX_SIZE, String.valueOf(CONSTANT_FN_FOLDING_MAX_SIZE_DEFAULT)));
            //Apply the optimization:
            DifferentialFunction df = op.getOp();
            df.clearArrays();
            for (int i = 0; i < in.size(); i++) {
                String s = in.get(i);
                INDArray arr = sd.getVariable(s).getArr();
                if (df instanceof CustomOp) {
                    ((CustomOp) df).addInputArgument(arr);
                } else {
                    if (i == 0)
                        ((Op) df).setX(arr);
                    else
                        ((Op) df).setY(arr);
                }
            }

            INDArray[] outputs;
            if (df instanceof CustomOp) {
                CustomOp o = (CustomOp) df;
                Nd4j.exec(o);
                outputs = new INDArray[o.numOutputArguments()];
                for (int j = 0; j < outputs.length; j++) {
                    outputs[j] = o.getOutputArgument(j);
                }
            } else {
                Op o = (Op) df;
                Nd4j.exec(o);
                outputs = new INDArray[]{o.z()};
            }
            long sizeCount = 0;
            for (INDArray i : outputs) {
                if (!i.dataType().isNumerical())
                    continue;
                sizeCount += i.length() * i.dataType().width();
            }

            if (sizeCount > maxSizeToApply)
                return false;

            //Convert outputs to constants
            List<String> outputNames = op.getOutputsOfOp();
            for(int i=0; i<outputNames.size(); i++ ){
                String n = outputNames.get(i);
                sd.getVariable(n).setVariableType(VariableType.CONSTANT);
                constantArrays.setArray(n, outputs[i]);
                sd.getVariables().get(n).setOutputOfOp(null);
            }

            //Remove the op: TODO Make util method?
            sd.getOps().remove(df.getOwnName());
            for(String s : op.getInputsToOp()){
                Variable v = sd.getVariables().get(s);
                v.getInputsForOp().remove(op.getName());
            }
            return true;
        }
    }
}
