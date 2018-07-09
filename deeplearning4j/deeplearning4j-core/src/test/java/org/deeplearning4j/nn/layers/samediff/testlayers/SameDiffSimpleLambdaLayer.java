package org.deeplearning4j.nn.layers.samediff.testlayers;

import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

public class SameDiffSimpleLambdaLayer extends SameDiffLambdaLayer {
    @Override
    public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput) {
        return layerInput.add(1.0).mul(2.0);
    }
}
