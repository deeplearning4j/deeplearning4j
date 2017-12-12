package org.deeplearning4j.nn.conf.layers.recurrent;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.params.BidirectionalParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

@NoArgsConstructor
@Data
public class Bidirectional extends Layer {

    public enum Mode {SUM, MUL, AVERAGE, CONCAT}

    private Layer underlying;
    private Mode mode;
    private BidirectionalParamInitializer initializer;

    public Bidirectional(@NonNull Mode mode, @NonNull Layer layer){
        if(!(layer instanceof BaseRecurrentLayer)){
            throw new IllegalArgumentException("Cannot wrap a non-recurrent layer: config must extend BaseRecurrentLayer. " +
                    "Got class: " + layer.getClass());
        }
        this.underlying = layer;
        this.mode = mode;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<IterationListener> iterationListeners, int layerIndex,
                                                       INDArray layerParamsView, boolean initializeParams) {
        return null;
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
        InputType outOrig = underlying.getOutputType(layerIndex, inputType);
        InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent)outOrig;
        if(mode == Mode.CONCAT){
            return InputType.recurrent(2 * r.getSize());
        } else {
            return r;
        }
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        underlying.setNIn(inputType, override);
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return underlying.getPreProcessorForInputType(inputType);
    }

    @Override
    public double getL1ByParam(String paramName) {
        //Strip "F_" or "R_" from param name
        return underlying.getL1ByParam(paramName.substring(2));
    }

    @Override
    public double getL2ByParam(String paramName) {
        return underlying.getL2ByParam(paramName.substring(2));
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return underlying.isPretrainParam(paramName.substring(2));
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        LayerMemoryReport lmr = underlying.getMemoryReport(inputType);
        //TODO double everything...
        return lmr;
    }
}
