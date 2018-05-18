package org.deeplearning4j.nn.params;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.samediff.AbstractSameDiffLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

@Slf4j
public class SameDiffParamInitializer implements ParamInitializer {

    private static final SameDiffParamInitializer INSTANCE = new SameDiffParamInitializer();

    public static SameDiffParamInitializer getInstance() {
        return INSTANCE;
    }

    @Override
    public int numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public int numParams(Layer layer) {
        AbstractSameDiffLayer sd = (AbstractSameDiffLayer)layer;
        Map<String,int[]> m = sd.getLayerParams().getParamShapes();
        int n = 0;
        for(int[] arr : m.values()){
            n += ArrayUtil.prod(arr);
        }
        return n;
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        AbstractSameDiffLayer sd = (AbstractSameDiffLayer)layer;
        return sd.getLayerParams().getParameterKeys();
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        AbstractSameDiffLayer sd = (AbstractSameDiffLayer)layer;
        return sd.getLayerParams().getWeightParameterKeys();
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        AbstractSameDiffLayer sd = (AbstractSameDiffLayer)layer;
        return sd.getLayerParams().getBiasParameterKeys();
    }

    @Override
    public boolean isWeightParam(Layer layer, String key) {
        return weightKeys(layer).contains(key);
    }

    @Override
    public boolean isBiasParam(Layer layer, String key) {
        return biasKeys(layer).contains(key);
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        AbstractSameDiffLayer sd = (AbstractSameDiffLayer) conf.getLayer();
        Map<String,INDArray> out = subsetAndReshape(sd.getLayerParams().getParameterKeys(),
                sd.getLayerParams().getParamShapes(), paramsView, sd);
        if(initializeParams){
            sd.initializeParameters(out);
        }

        for(String s : sd.getLayerParams().getParameterKeys()){
            conf.addVariable(s);
        }

        return out;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        AbstractSameDiffLayer sd = (AbstractSameDiffLayer) conf.getLayer();
        return subsetAndReshape(sd.getLayerParams().getParameterKeys(), sd.getLayerParams().getParamShapes(),
                gradientView, sd);
    }

    private Map<String,INDArray> subsetAndReshape(List<String> params, Map<String,int[]> paramShapes, INDArray view,
                                                  AbstractSameDiffLayer sdl){
        Map<String,INDArray> out = new LinkedHashMap<>();
        int soFar = 0;
        for(String s : params){
            int[] sh = paramShapes.get(s);
            int length = ArrayUtil.prod(sh);
            INDArray sub = view.get(point(0), interval(soFar, soFar + length));

            // FIXME: int cast
            if(!Arrays.equals(ArrayUtil.toInts(sub.shape()), sh)){
                sub = sub.reshape(sdl.paramReshapeOrder(s), sh); //TODO do we want to allow users to override initialization order?
            }
            out.put(s, sub);

            soFar += length;
        }
        return out;
    }
}
