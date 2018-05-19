package org.deeplearning4j.nn.conf.layers;

import lombok.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;

@Data
@NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public abstract class BaseOutputLayer extends FeedForwardLayer {

    protected ILossFunction lossFn;
    protected boolean hasBias = true;

    protected BaseOutputLayer(Builder builder) {
        super(builder);
        this.lossFn = builder.lossFn;
        this.hasBias = builder.hasBias;
    }

    public boolean hasBias(){
        return hasBias;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        //Basically a dense layer...
        InputType outputType = getOutputType(-1, inputType);

        val numParams = initializer().numParams(this);
        val updaterStateSize = (int) getIUpdater().stateSize(numParams);

        int trainSizeFixed = 0;
        int trainSizeVariable = 0;
        if (getIDropout() != null) {
            if (false) {
                //TODO drop connect
                //Dup the weights... note that this does NOT depend on the minibatch size...
                trainSizeVariable += 0; //TODO
            } else {
                //Assume we dup the input
                trainSizeVariable += inputType.arrayElementsPerExample();
            }
        }

        //Also, during backprop: we do a preOut call -> gives us activations size equal to the output size
        // which is modified in-place by activation function backprop
        // then we have 'epsilonNext' which is equivalent to input size
        trainSizeVariable += outputType.arrayElementsPerExample();

        return new LayerMemoryReport.Builder(layerName, OutputLayer.class, inputType, outputType)
                        .standardMemory(numParams, updaterStateSize)
                        .workingMemory(0, 0, trainSizeFixed, trainSizeVariable) //No additional memory (beyond activations) for inference
                        .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching
                        .build();
    }


    public static abstract class Builder<T extends Builder<T>> extends FeedForwardLayer.Builder<T> {
        protected ILossFunction lossFn = new LossMCXENT();
        private boolean hasBias = true;

        public Builder() {}

        public Builder(LossFunction lossFunction) {
            lossFunction(lossFunction);
        }

        public Builder(ILossFunction lossFunction) {
            this.lossFn = lossFunction;
        }

        public T lossFunction(LossFunction lossFunction) {
            return lossFunction(lossFunction.getILossFunction());
        }

        /**
         * If true (default): include bias parameters in the model. False: no bias.
         *
         * @param hasBias If true: include bias parameters in this model
         */
        public T hasBias(boolean hasBias){
            this.hasBias = hasBias;
            return (T)this;
        }

        public T lossFunction(ILossFunction lossFunction) {
            this.lossFn = lossFunction;
            return (T) this;
        }
    }
}
