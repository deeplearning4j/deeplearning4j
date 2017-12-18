package org.deeplearning4j.nn.layers.samediff;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.samediff.BaseSameDiffLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.AbstractLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.util.HashMap;
import java.util.Map;

public class SameDiffLayer extends AbstractLayer<BaseSameDiffLayer> {

    private static final String INPUT_KEY = "input";

    protected SameDiff sameDiff;
    protected String outputKey;


    public SameDiffLayer(NeuralNetConfiguration conf){
        super(conf);
    }



    @Override
    public Layer clone() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public void clearNoiseWeightParams() {
        //TODO - properly support noise weight...
    }

    @Override
    public INDArray activate(boolean training) {
        if(sameDiff == null){
            doInit();
        }

        SameDiff sd = sameDiff.getFunction(outputKey);
        //Build map:
        Map<String, INDArray> map = new HashMap<>(paramTable());
        map.put(INPUT_KEY, input);

        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            return sd.execAndEndResult();
        }
    }

    @Override
    public INDArray preOutput(boolean training) {
        return activate(training);
    }


    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        Gradient g = new DefaultGradient();

        SameDiff sd = sameDiff.getFunction(outputKey);
        INDArray dLdIn;
        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()){
            sd.execBackwards();
            for(String s : layerConf().paramKeys() ){
                INDArray pg = sd.grad(s).getArr();
                g.gradientForVariable().put(s, pg);
            }

            dLdIn = sd.grad(INPUT_KEY).getArr();
        }

        return new Pair<>(g, dLdIn);
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        double l2Sum = 0.0;
        for (Map.Entry<String, INDArray> entry : paramTable().entrySet()) {
            double l2 = conf.getL2ByParam(entry.getKey());
            if (l2 > 0) {
                double norm2 = getParam(entry.getKey()).norm2Number().doubleValue();
                l2Sum += 0.5 * l2 * norm2 * norm2;
            }
        }

        return l2Sum;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        double l1Sum = 0.0;
        for (Map.Entry<String, INDArray> entry : paramTable().entrySet()) {
            double l1 = conf.getL1ByParam(entry.getKey());
            if (l1 > 0) {
                double norm1 = getParam(entry.getKey()).norm1Number().doubleValue();
                l1Sum += l1 * norm1;
            }
        }

        return l1Sum;
    }

    protected void doInit(){
        sameDiff = SameDiff.create();
        Map<String,INDArray > p = paramTable();

        int[] inputShape = input.shape().clone();
        inputShape[0] = -1;
        SDVariable inputVar = sameDiff.var(INPUT_KEY, inputShape);     //TODO WHAT ABOUT VARIABLE SIZES?
        layerConf().defineLayer(sameDiff, inputVar, p);
    }
}
