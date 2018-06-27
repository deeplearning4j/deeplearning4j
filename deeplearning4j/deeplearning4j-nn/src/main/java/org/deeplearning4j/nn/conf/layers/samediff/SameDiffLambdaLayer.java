package org.deeplearning4j.nn.conf.layers.samediff;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public abstract class SameDiffLambdaLayer extends BaseSameDiffLayer {


    public abstract SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput);

    @Override
    public List<SDVariable> defineLayer(SameDiff sameDiff, SDVariable layerInput, Map<String, SDVariable> paramTable) {
        return Collections.singletonList(defineLayer(sameDiff, layerInput));
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        //TODO let's try to infer output shape from input shape, using SameDiff + DefineLayer
        throw new UnsupportedOperationException("Override SameDiffLamdaLayer.getOutputType to use OutputType functionality";)
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        //No op: lambda layer doesn't have parameters
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        //No op: lambda layer doesn't have parameters
    }
}
