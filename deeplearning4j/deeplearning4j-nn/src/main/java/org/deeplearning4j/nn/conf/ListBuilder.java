package org.deeplearning4j.nn.conf;

import lombok.Data;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataType;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Fluent interface for building a list of configurations
 */
@Slf4j
@Data
public class ListBuilder extends BaseBuilder {
    private int layerCounter = -1; //Used only for .layer(Layer) method
    private Map<Integer, NeuralNetConfiguration.Builder> layerwise;
    private NeuralNetConfiguration.Builder globalConfig;

    // Constructor
    public ListBuilder(NeuralNetConfiguration.Builder globalConfig, Map<Integer, NeuralNetConfiguration.Builder> layerMap) {
        super();
        this.globalConfig = globalConfig;
        this.layerwise = layerMap;
    }

    public ListBuilder(NeuralNetConfiguration.Builder globalConfig) {
        this(globalConfig, new HashMap<>());
    }

    public ListBuilder layer(int ind, @NonNull Layer layer) {
        if (layerwise.containsKey(ind)) {
            log.info("Layer index {} already exists, layer of type {} will be replace by layer type {}",
                    ind, layerwise.get(ind).getClass().getSimpleName(), layer.getClass().getSimpleName());
            layerwise.get(ind).layer(layer);
        } else {
            layerwise.put(ind, globalConfig.clone().layer(layer));
        }
        if (layerCounter < ind) {
            //Edge case: user is mixing .layer(Layer) and .layer(int, Layer) calls
            //This should allow a .layer(A, X) and .layer(Y) to work such that layer Y is index (A+1)
            layerCounter = ind;
        }
        return this;
    }

    public ListBuilder layer(Layer layer) {
        return layer(++layerCounter, layer);
    }

    public Map<Integer, NeuralNetConfiguration.Builder> getLayerwise() {
        return layerwise;
    }

    @Override
    public ListBuilder overrideNinUponBuild(boolean overrideNinUponBuild) {
        super.overrideNinUponBuild(overrideNinUponBuild);
        return this;
    }

    @Override
    public ListBuilder inputPreProcessor(Integer layer, InputPreProcessor processor) {
        super.inputPreProcessor(layer, processor);
        return this;
    }



    @Override
    public ListBuilder cacheMode(@NonNull CacheMode cacheMode) {
        super.cacheMode(cacheMode);
        return this;
    }



    @Override
    public ListBuilder tBPTTLength(int bpttLength) {
        super.tBPTTLength(bpttLength);
        return this;
    }

    @Override
    public ListBuilder tBPTTForwardLength(int forwardLength) {
        super.tBPTTForwardLength(forwardLength);
        return this;
    }

    @Override
    public ListBuilder tBPTTBackwardLength(int backwardLength) {
        super.tBPTTBackwardLength(backwardLength);
        return this;
    }


    @Override
    public ListBuilder validateOutputLayerConfig(boolean validate) {
        super.validateOutputLayerConfig(validate);
        return this;
    }

    @Override
    public ListBuilder validateTbpttConfig(boolean validate) {
        super.validateTbpttConfig(validate);
        return this;
    }

    @Override
    public ListBuilder dataType(@NonNull DataType dataType) {
        super.dataType(dataType);
        return this;
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
    }

    @Override
    public ListBuilder setInputType(InputType inputType) {
        return (ListBuilder) super.setInputType(inputType);
    }

    /**
     * A convenience method for setting input types: note that for example .inputType().convolutional(h,w,d)
     * is equivalent to .setInputType(InputType.convolutional(h,w,d))
     */
    public InputTypeBuilder inputType() {
        return new InputTypeBuilder();
    }

    /**
     * For the (perhaps partially constructed) network configuration, return a list of activation sizes for each
     * layer in the network.<br>
     * Note: To use this method, the network input type must have been set using {@link #setInputType(InputType)} first
     *
     * @return A list of activation types for the network, indexed by layer number
     */
    public List<InputType> getLayerActivationTypes() {
        Preconditions.checkState(inputType != null, "Can only calculate activation types if input type has" +
                "been set. Use setInputType(InputType)");

        MultiLayerConfiguration conf;
        try {
            conf = build();
        } catch (Exception e) {
            throw new RuntimeException("Error calculating layer activation types: error instantiating MultiLayerConfiguration", e);
        }

        return conf.getLayerActivationTypes(inputType);
    }

    /**
     * Build the multi layer network
     * based on this neural network and
     * overr ridden parameters
     *
     * @return the configuration to build
     */
    public MultiLayerConfiguration build() {
        List<NeuralNetConfiguration> list = new ArrayList<>();
        if (layerwise.isEmpty())
            throw new IllegalStateException("Invalid configuration: no layers defined");
        for (int i = 0; i < layerwise.size(); i++) {
            if (layerwise.get(i) == null) {
                throw new IllegalStateException("Invalid configuration: layer number " + i
                        + " not specified. Expect layer " + "numbers to be 0 to " + (layerwise.size() - 1)
                        + " inclusive (number of layers defined: " + layerwise.size() + ")");
            }
            if (layerwise.get(i).getLayer() == null)
                throw new IllegalStateException("Cannot construct network: Layer config for" + "layer with index "
                        + i + " is not defined)");

            //Layer names: set to default, if not set
            if (layerwise.get(i).getLayer().getLayerName() == null) {
                layerwise.get(i).getLayer().setLayerName("layer" + i);
            }

            list.add(layerwise.get(i).build());
        }

        WorkspaceMode wsmTrain = (globalConfig.setTWM ? globalConfig.trainingWorkspaceMode : trainingWorkspaceMode);
        WorkspaceMode wsmTest = (globalConfig.setIWM ? globalConfig.inferenceWorkspaceMode : inferenceWorkspaceMode);


        MultiLayerConfiguration.Builder builder = new MultiLayerConfiguration.Builder().inputPreProcessors(inputPreProcessors)
                .backpropType(backpropType).tBPTTForwardLength(tbpttFwdLength)
                .tBPTTBackwardLength(tbpttBackLength).setInputType(this.inputType)
                .trainingWorkspaceMode(wsmTrain).cacheMode(globalConfig.cacheMode)
                .inferenceWorkspaceMode(wsmTest).confs(list).validateOutputLayerConfig(validateOutputConfig)
                .overrideNinUponBuild(overrideNinUponBuild)
                .dataType(globalConfig.dataType);
        return builder.build();
    }

    /**
     * Helper class for setting input types
     */
    public class InputTypeBuilder {
        /**
         * See {@link InputType#convolutional(long, long, long)}
         */
        public ListBuilder convolutional(int height, int width, int depth) {
            return ListBuilder.this.setInputType(InputType.convolutional(height, width, depth));
        }

        /**
         * * See {@link InputType#convolutionalFlat(long, long, long)}
         */
        public ListBuilder convolutionalFlat(int height, int width, int depth) {
            return ListBuilder.this.setInputType(InputType.convolutionalFlat(height, width, depth));
        }

        /**
         * See {@link InputType#feedForward(long)}
         */
        public ListBuilder feedForward(int size) {
            return ListBuilder.this.setInputType(InputType.feedForward(size));
        }

        /**
         * See {@link InputType#recurrent(long)}}
         */
        public ListBuilder recurrent(int size) {
            return ListBuilder.this.setInputType(InputType.recurrent(size));
        }
    }
}
