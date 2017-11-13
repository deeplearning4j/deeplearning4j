package org.deeplearning4j.nn.conf;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *  Interface for building a list of configurations

 *
 * @author Max Pumperla
 */
@Slf4j
public class SequentialConfiguration extends NeuralNetConfiguration {


    public static class ListBuilder extends MultiLayerConfiguration.Builder {
        private int layerCounter = -1; //Used only for .layer(Layer) method
        private Map<Integer, Layer> layerwise;
        private Builder globalConfig;

        // Constructor
        public ListBuilder(Builder globalConfig, Map<Integer, Layer> layerMap) {
            this.globalConfig = globalConfig;
            this.layerwise = layerMap;
        }

        public ListBuilder(Builder globalConfig) {
            this(globalConfig, new HashMap<Integer, Layer>());
        }

        public ListBuilder backprop(boolean backprop) {
            this.backprop = backprop;

            return this;
        }

        public ListBuilder pretrain(boolean pretrain) {
            this.pretrain = pretrain;
            return this;
        }

        public ListBuilder layer(int ind, @NonNull Layer layer) {
            if (layerwise.containsKey(ind)) {
                log.info("Layer index {} already exists, layer of type {} will be replace by layer type {}",
                        ind, layerwise.get(ind).getClass().getSimpleName(), layer.getClass().getSimpleName());
                layerwise.put(ind, layer);
            } else {
                layerwise.put(ind, layer);
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

        public Map<Integer, Layer> getLayerwise() {
            return layerwise;
        }

        @Override
        public ListBuilder setInputType(InputType inputType) {
            return (ListBuilder) super.setInputType(inputType);
        }

        /**
         * A convenience method for setting input types: note that for example .inputType().convolutional(h,w,d)
         * is equivalent to .setInputType(InputType.convolutional(h,w,d))
         */
        public ListBuilder.InputTypeBuilder inputType() {
            return new ListBuilder.InputTypeBuilder();
        }

        /**
         * Build the multi layer network
         * based on this neural network and
         * overridden parameters
         *
         * @return the configuration to build
         */
        public MultiLayerConfiguration build() {
            List<Layer> list = new ArrayList<>();
            if (layerwise.isEmpty())
                throw new IllegalStateException("Invalid configuration: no layers defined");
            for (int i = 0; i < layerwise.size(); i++) {
                if (layerwise.get(i) == null) {
                    throw new IllegalStateException("Invalid configuration: layer number " + i
                            + " not specified. Expect layer " + "numbers to be 0 to " + (layerwise.size() - 1)
                            + " inclusive (number of layers defined: " + layerwise.size() + ")");
                }
                if (layerwise.get(i) == null)
                    throw new IllegalStateException("Cannot construct network: Layer config for" + "layer with index "
                            + i + " is not defined)");

                //Layer names: set to default, if not set
                if (layerwise.get(i).getLayerName() == null) {
                    layerwise.get(i).setLayerName("layer" + i);
                }

                list.add(layerwise.get(i));
            }
            return new MultiLayerConfiguration.Builder()
                    .globalConfiguration(globalConfig.globalConf)
                    .backprop(backprop).inputPreProcessors(inputPreProcessors)
                    .pretrain(pretrain).backpropType(backpropType).tBPTTForwardLength(tbpttFwdLength)
                    .tBPTTBackwardLength(tbpttBackLength).setInputType(this.inputType)
                    .trainingWorkspaceMode(globalConfig.globalConf.getTrainingWorkspaceMode())
                    .cacheMode(globalConfig.globalConf.getCacheMode())
                    .inferenceWorkspaceMode(globalConfig.globalConf.getInferenceWorkspaceMode()).confs(list).build();
        }

        /**
         * Helper class for setting input types
         */
        public class InputTypeBuilder {
            /**
             * See {@link InputType#convolutional(int, int, int)}
             */
            public ListBuilder convolutional(int height, int width, int depth) {
                return ListBuilder.this.setInputType(InputType.convolutional(height, width, depth));
            }

            /**
             * * See {@link InputType#convolutionalFlat(int, int, int)}
             */
            public ListBuilder convolutionalFlat(int height, int width, int depth) {
                return ListBuilder.this.setInputType(InputType.convolutionalFlat(height, width, depth));
            }

            /**
             * See {@link InputType#feedForward(int)}
             */
            public ListBuilder feedForward(int size) {
                return ListBuilder.this.setInputType(InputType.feedForward(size));
            }

            /**
             * See {@link InputType#recurrent(int)}}
             */
            public ListBuilder recurrent(int size) {
                return ListBuilder.this.setInputType(InputType.recurrent(size));
            }
        }
    }



}
