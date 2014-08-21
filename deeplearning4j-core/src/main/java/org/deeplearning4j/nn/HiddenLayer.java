package org.deeplearning4j.nn;


import java.io.Serializable;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.linalg.api.activation.ActivationFunction;
import org.deeplearning4j.linalg.api.activation.Activations;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.transforms.Transforms;


/**
 * Vectorized Hidden Layer
 * @author Adam Gibson
 *
 */
public class HiddenLayer implements Serializable {

    protected static final long serialVersionUID = 915783367350830495L;
    protected int nIn;
    protected int nOut;
    protected INDArray W;
    protected INDArray b;
    protected RandomGenerator rng;
    protected INDArray input;
    protected ActivationFunction activationFunction = Activations.sigmoid();
    protected RealDistribution dist;
    protected boolean concatBiases = false;
    protected WeightInit weightInit;
    protected HiddenLayer() {}

    public HiddenLayer(int nIn, int nOut, INDArray W, INDArray b, RandomGenerator rng,INDArray input,ActivationFunction activationFunction) {
        this(nIn,nOut,W,b,rng,input,activationFunction,null);
    }


    public HiddenLayer(int nIn, int nOut, INDArray W, INDArray b, RandomGenerator rng,INDArray input) {
        this(nIn,nOut,W,b,rng,input,null,null,null);
    }
    public HiddenLayer(int nIn, int nOut, INDArray W, INDArray b, RandomGenerator rng,INDArray input,WeightInit weightInit) {
        this(nIn,nOut,W,b,rng,input,null,null,weightInit);
    }



    public HiddenLayer(int nIn, int nOut, INDArray W, INDArray b, RandomGenerator rng,INDArray input,ActivationFunction activationFunction,RealDistribution dist,WeightInit weightInit) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.input = input;
        if(activationFunction != null)
            this.activationFunction = activationFunction;

        if(rng == null) {
            this.rng = new MersenneTwister(1234);
        }
        else
            this.rng = rng;

        if(dist == null)
            this.dist = new NormalDistribution(this.rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
        else
            this.dist = dist;

        if(W == null) {

            this.W = NDArrays.zeros(nIn, nOut);

            for(int i = 0; i < this.W.rows(); i++)
                this.W.putRow(i,NDArrays.create(this.dist.sample(this.W.columns())));
        }

        else
            this.W = W;


        if(b == null)
            this.b = NDArrays.zeros(nOut);
        else
            this.b = b;
    }


    public HiddenLayer(int nIn, int nOut, INDArray W, INDArray b, RandomGenerator rng,INDArray input,RealDistribution dist,WeightInit weightInit) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.input = input;


        if(rng == null)
            this.rng = new MersenneTwister(1234);

        else
            this.rng = rng;

        if(dist == null)
            this.dist = new NormalDistribution(this.rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
        else
            this.dist = dist;

        if(W == null) {
            if(weightInit != null) {
                this.W = WeightInitUtil.initWeights(nIn,nOut,this.weightInit,activationFunction);
            }
            else {
                this.W = NDArrays.zeros(nIn,nOut);

                for(int i = 0; i < this.W.rows(); i++)
                    this.W.putRow(i,NDArrays.create(this.dist.sample(this.W.columns())));

            }
        }

        else
            this.W = W;


        if(b == null)
            this.b = NDArrays.zeros(nOut);
        else
            this.b = b;
    }



    public HiddenLayer(int nIn, int nOut, INDArray W, INDArray b, RandomGenerator rng,INDArray input,ActivationFunction activationFunction,RealDistribution dist) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.input = input;
        if(activationFunction != null)
            this.activationFunction = activationFunction;

        if(rng == null) {
            this.rng = new MersenneTwister(1234);
        }
        else
            this.rng = rng;

        if(dist == null)
            this.dist = new NormalDistribution(this.rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
        else
            this.dist = dist;

        if(W == null) {

            this.W = NDArrays.zeros(nIn,nOut);

            for(int i = 0; i < this.W.rows(); i++)
                this.W.putRow(i,NDArrays.create(this.dist.sample(this.W.columns())));
        }

        else
            this.W = W;


        if(b == null)
            this.b = NDArrays.zeros(nOut);
        else
            this.b = b;
    }


    public HiddenLayer(int nIn, int nOut, INDArray W, INDArray b, RandomGenerator rng,INDArray input,RealDistribution dist) {
        this.nIn = nIn;
        this.nOut = nOut;
        this.input = input;


        if(rng == null)
            this.rng = new MersenneTwister(1234);

        else
            this.rng = rng;

        if(dist == null)
            this.dist = new NormalDistribution(this.rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);
        else
            this.dist = dist;

        if(W == null) {
            if(weightInit != null) {
                this.W = WeightInitUtil.initWeights(nIn,nOut,this.weightInit,activationFunction);
            }
            else {
                this.W = NDArrays.zeros(nIn,nOut);

                for(int i = 0; i < this.W.rows(); i++)
                    this.W.putRow(i,NDArrays.create(this.dist.sample(this.W.columns())));

            }
        }

        else
            this.W = W;


        if(b == null)
            this.b = NDArrays.zeros(nOut);
        else
            this.b = b;
    }

    public WeightInit getWeightInit() {
        return weightInit;
    }

    public void setWeightInit(WeightInit weightInit) {
        this.weightInit = weightInit;
    }

    public  int getnIn() {
        return nIn;
    }

    public  void setnIn(int nIn) {
        this.nIn = nIn;
    }

    public  int getnOut() {
        return nOut;
    }

    public  void setnOut(int nOut) {
        this.nOut = nOut;
    }

    public  INDArray getW() {
        return W;
    }

    public  void setW(INDArray w) {
        W = w;
    }

    public  INDArray getB() {
        return b;
    }

    public  void setB(INDArray b) {
        this.b = b;
    }

    public  RandomGenerator getRng() {
        return rng;
    }

    public  void setRng(RandomGenerator rng) {
        this.rng = rng;
    }

    public  INDArray getInput() {
        return input;
    }

    public  void setInput(INDArray input) {
        this.input = input;
    }

    public  ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public  void setActivationFunction(
            ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public boolean isConcatBiases() {
        return concatBiases;
    }

    public void setConcatBiases(boolean concatBiases) {
        this.concatBiases = concatBiases;
    }

    @Override
    public HiddenLayer clone() {
        HiddenLayer layer = new HiddenLayer();
        layer.b = b.dup();
        layer.W = W.dup();
        if(input != null)
            layer.input = input.dup();
        if(dist != null)
            layer.dist = dist;
        layer.activationFunction = activationFunction;
        layer.nOut = nOut;
        layer.weightInit = weightInit;
        layer.nIn = nIn;
        layer.rng = rng;
        return layer;
    }

    /**
     * Returns a transposed version of this hidden layer.
     * A transpose is just the bias and weights flipped
     * + number of ins and outs flipped
     * @return the transposed version of this hidden layer
     */
    public HiddenLayer transpose() {
        HiddenLayer layer = new HiddenLayer();
        layer.b = b.dup();
        layer.W = W.transpose();
        if(input != null)
            layer.input = input.transpose();
        if(dist != null)
            layer.dist = dist;
        layer.activationFunction = activationFunction;
        layer.nOut = nIn;
        layer.weightInit = weightInit;
        layer.nIn = nOut;
        layer.concatBiases = concatBiases;
        layer.rng = rng;
        return layer;
    }



    /**
     * Trigger an activation with the last specified input
     * @return the activation of the last specified input
     */
    public  INDArray activate() {
        INDArray activation =  getActivationFunction().apply(getInput().mmul(getW()).addRowVector(getB()));
        return activation;
    }

    /**
     * Initialize the layer with the given input
     * and return the activation for this layer
     * given this input
     * @param input the input to use
     * @return
     */
    public  INDArray activate(INDArray input) {
        if(input != null)
            this.input = Transforms.stabilize(input.dup(), 1);
        return activate();
    }


    public static class Builder {
        protected int nIn;
        protected int nOut;
        protected INDArray W;
        protected INDArray b;
        protected RandomGenerator rng;
        protected INDArray input;
        protected ActivationFunction activationFunction = Activations.sigmoid();
        protected RealDistribution dist;
        protected boolean concatBiases = false;
        protected WeightInit weightInit;


        public Builder weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return this;
        }

        public Builder concatBiases(boolean concatBiases) {
            this.concatBiases = concatBiases;
            return this;
        }

        public Builder dist(RealDistribution dist) {
            this.dist = dist;
            return this;
        }

        public Builder nIn(int nIn) {
            this.nIn = nIn;
            return this;
        }

        public Builder nOut(int nOut) {
            this.nOut = nOut;
            return this;
        }

        public Builder withWeights(INDArray W) {
            this.W = W;
            return this;
        }

        public Builder withRng(RandomGenerator gen) {
            this.rng = gen;
            return this;
        }

        public Builder withActivation(ActivationFunction function) {
            this.activationFunction = function;
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

        public HiddenLayer build() {
            HiddenLayer ret =  new HiddenLayer(nIn,nOut,W,b,rng,input,weightInit);
            ret.weightInit = weightInit;
            ret.activationFunction = activationFunction;
            ret.concatBiases = concatBiases;
            ret.dist = dist;
            return ret;
        }

    }

}