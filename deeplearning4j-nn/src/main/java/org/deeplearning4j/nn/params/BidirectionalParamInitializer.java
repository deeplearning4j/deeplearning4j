package org.deeplearning4j.nn.params;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class BidirectionalParamInitializer implements ParamInitializer {
    public static final String FORWARD_PREFIX = "F_";
    public static final String BACKWARD_PREFIX = "R_";

    private final Bidirectional layer;
    private final BaseRecurrentLayer underlying;

    private List<String> paramKeys;
    private List<String> weightKeys;
    private List<String> biasKeys;

    public BidirectionalParamInitializer(Bidirectional layer){
        this.layer = layer;
        this.underlying = underlying(layer);
    }

    @Override
    public int numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public int numParams(Layer layer) {
        return 2 * underlying(layer).initializer().numParams(underlying(layer));
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        if(paramKeys == null) {
            BaseRecurrentLayer u = underlying(layer);
            List<String> orig = u.initializer().paramKeys(u);
            paramKeys = withPrefixes(orig);
        }
        return paramKeys;
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        if(weightKeys == null) {
            BaseRecurrentLayer u = underlying(layer);
            List<String> orig = u.initializer().weightKeys(u);
            weightKeys = withPrefixes(orig);
        }
        return weightKeys;
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        if(biasKeys == null) {
            BaseRecurrentLayer u = underlying(layer);
            List<String> orig = u.initializer().weightKeys(u);
            biasKeys = withPrefixes(orig);
        }
        return biasKeys;
    }

    @Override
    public boolean isWeightParam(String key) {
        return weightKeys(layer).contains(key);
    }

    @Override
    public boolean isBiasParam(String key) {
        return biasKeys(layer).contains(key);
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        int n = paramsView.length()/2;
        INDArray forwardView = paramsView.get(point(0), interval(0, n));
        INDArray backwardView = paramsView.get(point(0), interval(n, 2*n));

        Map<String, INDArray> origFwd = underlying.initializer().init(conf, forwardView, initializeParams);
        Map<String, INDArray> origBwd = underlying.initializer().init(conf, backwardView, initializeParams);

        Map<String,INDArray> out = new LinkedHashMap<>();
        for( Map.Entry<String, INDArray> e : origFwd.entrySet()){
            out.put(FORWARD_PREFIX + e.getKey(), e.getValue());
        }
        for( Map.Entry<String, INDArray> e : origBwd.entrySet()){
            out.put(BACKWARD_PREFIX + e.getKey(), e.getValue());
        }

        return out;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        int n = gradientView.length()/2;
        INDArray forwardView = gradientView.get(point(0), interval(0, n));
        INDArray backwardView = gradientView.get(point(0), interval(n, 2*n));

        Map<String, INDArray> origFwd = underlying.initializer().getGradientsFromFlattened(conf, forwardView);
        Map<String, INDArray> origBwd = underlying.initializer().getGradientsFromFlattened(conf, backwardView);

        Map<String,INDArray> out = new LinkedHashMap<>();
        for( Map.Entry<String, INDArray> e : origFwd.entrySet()){
            out.put(FORWARD_PREFIX + e.getKey(), e.getValue());
        }
        for( Map.Entry<String, INDArray> e : origBwd.entrySet()){
            out.put(BACKWARD_PREFIX + e.getKey(), e.getValue());
        }

        return out;
    }

    private BaseRecurrentLayer underlying(Layer layer){
        Bidirectional b = (Bidirectional)layer;
        return (BaseRecurrentLayer)b.getUnderlying();
    }

    private List<String> withPrefixes(List<String> orig){
        List<String> out = new ArrayList<>();
        for(String s : orig){
            out.add(FORWARD_PREFIX + s);
        }
        for(String s : orig){
            out.add(BACKWARD_PREFIX + s);
        }
        return out;
    }
}
