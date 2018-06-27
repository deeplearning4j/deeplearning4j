package org.deeplearning4j.nn.layers.samediff.testlayers;

import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaVertex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

public class SameDiffSimpleLambdaVertex extends SameDiffLambdaVertex {

    @Override
    public SDVariable defineVertex(SameDiff sameDiff, VertexInputs inputs) {
        SDVariable in1 = inputs.getInput(0);
        SDVariable in2 = inputs.getInput(1);
        return in1.mul(in2);
    }
}
