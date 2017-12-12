package org.deeplearning4j.nn.conf.layers.recurrent;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.api.layers.RecurrentLayer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.layers.recurrent.BidirectionalLayer;
import org.deeplearning4j.nn.params.BidirectionalParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;

import java.util.Collection;
import java.util.Map;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

@NoArgsConstructor
@Data
public class Bidirectional extends Layer {

    public enum Mode {ADD, MUL, AVERAGE, CONCAT}

    private Layer fwd;
    private Layer bwd;
    private Mode mode;
    private BidirectionalParamInitializer initializer;

    public Bidirectional(Layer layer){
        this( Mode.CONCAT, layer);
    }

    public Bidirectional(@NonNull Mode mode, @NonNull Layer layer){
        if(!(layer instanceof BaseRecurrentLayer)){
            throw new IllegalArgumentException("Cannot wrap a non-recurrent layer: config must extend BaseRecurrentLayer. " +
                    "Got class: " + layer.getClass());
        }
        this.fwd = layer;
        this.bwd = layer.clone();
        this.mode = mode;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<IterationListener> iterationListeners, int layerIndex,
                                                       INDArray layerParamsView, boolean initializeParams) {
        NeuralNetConfiguration c1 = conf.clone();
        NeuralNetConfiguration c2 = conf.clone();
        c1.setLayer(fwd);
        c2.setLayer(bwd);

        int n = layerParamsView.length() / 2;
        INDArray fp = layerParamsView.get(point(0), interval(0,n));
        INDArray bp = layerParamsView.get(point(0), interval(n, 2*n));
        org.deeplearning4j.nn.api.layers.RecurrentLayer f
                = (RecurrentLayer) fwd.instantiate(c1, iterationListeners, layerIndex, fp, initializeParams);

        org.deeplearning4j.nn.api.layers.RecurrentLayer b
                = (RecurrentLayer) bwd.instantiate(c2, iterationListeners, layerIndex, bp, initializeParams);

        BidirectionalLayer ret = new BidirectionalLayer(conf, f, b);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);

        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        if(initializer == null){
            initializer = new BidirectionalParamInitializer(this);
        }
        return initializer;
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        InputType outOrig = fwd.getOutputType(layerIndex, inputType);
        InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent)outOrig;
        if(mode == Mode.CONCAT){
            return InputType.recurrent(2 * r.getSize());
        } else {
            return r;
        }
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        fwd.setNIn(inputType, override);
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return fwd.getPreProcessorForInputType(inputType);
    }

    @Override
    public double getL1ByParam(String paramName) {
        //Strip "F_" or "R_" from param name
        return fwd.getL1ByParam(paramName.substring(2));
    }

    @Override
    public double getL2ByParam(String paramName) {
        return fwd.getL2ByParam(paramName.substring(2));
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return fwd.isPretrainParam(paramName.substring(2));
    }

    /**
     * Get the updater for the given parameter. Typically the same updater will be used for all updaters, but this
     * is not necessarily the case
     *
     * @param paramName    Parameter name
     * @return             IUpdater for the parameter
     */
    public IUpdater getUpdaterByParam(String paramName) {
        String sub = paramName.substring(2);
        return fwd.getUpdaterByParam(sub);
    }

    @Override
    public void setLayerName(String layerName){
        fwd.setLayerName(layerName);
        bwd.setLayerName(layerName);
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        LayerMemoryReport lmr = fwd.getMemoryReport(inputType);
        //TODO double everything...
        return lmr;
    }
}
