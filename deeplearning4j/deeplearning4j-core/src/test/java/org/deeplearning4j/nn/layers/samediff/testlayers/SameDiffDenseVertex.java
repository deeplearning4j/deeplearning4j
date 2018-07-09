package org.deeplearning4j.nn.layers.samediff.testlayers;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.conf.layers.samediff.SDVertexParams;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffVertex;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

@NoArgsConstructor
@Data
public class SameDiffDenseVertex extends SameDiffVertex {

    private int nIn;
    private int nOut;
    private Activation activation;
    private WeightInit weightInit;

    public SameDiffDenseVertex(int nIn, int nOut, Activation activation, WeightInit weightInit){
        this.nIn = nIn;
        this.nOut = nOut;
        this.activation = activation;
        this.weightInit = weightInit;
    }

    @Override
    public SDVariable defineVertex(SameDiff sameDiff, Map<String, SDVariable> layerInput, Map<String, SDVariable> paramTable) {
        SDVariable weights = paramTable.get(DefaultParamInitializer.WEIGHT_KEY);
        SDVariable bias = paramTable.get(DefaultParamInitializer.BIAS_KEY);

        SDVariable mmul = sameDiff.mmul("mmul", layerInput.get("in"), weights);
        SDVariable z = mmul.add("z", bias);
        return activation.asSameDiff("out", sameDiff, z);
    }

    @Override
    public void defineParametersAndInputs(SDVertexParams params) {
        params.defineInputs("in");
        params.addWeightParam("W", nIn, nOut);
        params.addBiasParam("b", 1, nOut);
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        //Normally use 'c' order, but use 'f' for direct comparison to DL4J DenseLayer
        WeightInitUtil.initWeights(nIn, nOut, new long[]{nIn, nOut}, weightInit, null, 'f', params.get("W"));
        params.get("b").assign(0.0);
    }

    @Override
    public char paramReshapeOrder(String paramName){
        return 'f';     //To match DL4J DenseLayer - for easy comparison
    }
}
