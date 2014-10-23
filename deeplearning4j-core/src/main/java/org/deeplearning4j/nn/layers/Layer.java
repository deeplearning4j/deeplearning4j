package org.deeplearning4j.nn.layers;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;


/**
 * Vectorized Hidden Layer
 * @author Adam Gibson
 *
 */
public class Layer extends BaseLayer {

    protected static final long serialVersionUID = 915783367350830495L;

    public Layer(NeuralNetConfiguration conf, INDArray W, INDArray b, INDArray input) {
        super(conf, W, b, input);
    }



    public static class Builder {
        protected INDArray W;
        protected INDArray b;
        protected INDArray input;
        protected NeuralNetConfiguration conf;


        public Builder conf(NeuralNetConfiguration conf) {
            this.conf = conf;
            return this;
        }


        public Builder withWeights(INDArray W) {
            this.W = W;
            return this;
        }


        public Builder withBias(INDArray b) {
            this.b = b;
            return this;
        }

        public Builder withInput(INDArray input) {
            this.input = input;
            return this;
        }

        public Layer build() {
            Layer ret =  new Layer(conf,W,b,input);
            return ret;
        }

    }

}