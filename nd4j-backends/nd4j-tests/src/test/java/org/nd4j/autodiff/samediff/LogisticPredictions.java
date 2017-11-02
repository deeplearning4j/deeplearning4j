package org.nd4j.autodiff.samediff;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

public class LogisticPredictions implements SameDiff.SameDiffFunctionDefinition {
    /**
     * @param sameDiff
     * @param inputs
     * @param variableInputs
     * @return
     */
    @Override
    public SDVariable[] define(SameDiff sameDiff, Map<String, INDArray> inputs, SDVariable[] variableInputs) {
        SDVariable input = sameDiff.var("x",inputs.get("x"));
        SDVariable w = sameDiff.var("w",inputs.get("w"));
        SDVariable y = sameDiff.var("y",inputs.get("y"));
        SDVariable preOutput = sameDiff.mmul(input,w);
        SDVariable sigmoid = sameDiff.sigmoid(preOutput);

        return new SDVariable[]{sigmoid};
    }
}
