package org.deeplearning4j.nn.layers;


import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.linalg.api.activation.ActivationFunction;
import org.deeplearning4j.linalg.api.activation.Activations;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.transforms.Transforms;
import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.WeightInitUtil;


/**
 * Vectorized Hidden Layer
 * @author Adam Gibson
 *
 */
public class HiddenLayer implements org.deeplearning4j.nn.api.Layer {

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

    public HiddenLayer(int nIn, int nOut, INDArray W, INDArray b, RandomGenerator rng, INDArray input, ActivationFunction activationFunction) {
        this(nIn,nOut,W,b,rng,input,activationFunction,null);
    }


    public HiddenLayer(int nIn, int nOut, INDArray W, INDArray b, RandomGenerator rng, INDArray input) {
        this(nIn,nOut,W,b,rng,input,null,null,null);
    }
    public HiddenLayer(int nIn, int nOut, INDArray W, INDArray b, RandomGenerator rng, INDArray input, WeightInit weightInit) {
        this(nIn,nOut,W,b,rng,input,null,null,weightInit);
    }



    public HiddenLayer(int nIn, int nOut, INDArray W, INDArray b, RandomGenerator rng, INDArray input, ActivationFunction activationFunction, RealDistribution dist, WeightInit weightInit) {
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


    public HiddenLayer(int nIn, int nOut, INDArray W, INDArray b, RandomGenerator rng, INDArray input, RealDistribution dist, WeightInit weightInit) {
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
                this.W = WeightInitUtil.initWeights(nIn, nOut, this.weightInit, activationFunction);
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



    public HiddenLayer(int nIn, int nOut, INDArray W, INDArray b, RandomGenerator rng, INDArray input, ActivationFunction activationFunction, RealDistribution dist) {
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


    public HiddenLayer(int nIn, int nOut, INDArray W, INDArray b, RandomGenerator rng, INDArray input, RealDistribution dist) {
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

    @Override
    public WeightInit getWeightInit() {
        return weightInit;
    }

    @Override
    public void setWeightInit(WeightInit weightInit) {
        this.weightInit = weightInit;
    }

    @Override
    public  int getnIn() {
        return nIn;
    }

    @Override
    public  void setnIn(int nIn) {
        this.nIn = nIn;
    }

    @Override
    public  int getnOut() {
        return nOut;
    }

    @Override
    public  void setnOut(int nOut) {
        this.nOut = nOut;
    }

    @Override
    public  INDArray getW() {
        return W;
    }

    @Override
    public  void setW(INDArray w) {
        W = w;
    }

    @Override
    public  INDArray getB() {
        return b;
    }

    @Override
    public  void setB(INDArray b) {
        this.b = b;
    }

    @Override
    public  RandomGenerator getRng() {
        return rng;
    }

    @Override
    public  void setRng(RandomGenerator rng) {
        this.rng = rng;
    }

    @Override
    public  INDArray getInput() {
        return input;
    }

    @Override
    public  void setInput(INDArray input) {
        this.input = input;
    }

    @Override
    public  ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    @Override
    public  void setActivationFunction(
            ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    @Override
    public boolean isConcatBiases() {
        return concatBiases;
    }

    @Override
    public void setConcatBiases(boolean concatBiases) {
        this.concatBiases = concatBiases;
    }

    @Override
    public HiddenLayer clone() {
        HiddenLayer hiddeenLayer = new HiddenLayer();
        hiddeenLayer.b = b.dup();
        hiddeenLayer.W = W.dup();
        if(input != null)
            hiddeenLayer.input = input.dup();
        if(dist != null)
            hiddeenLayer.dist = dist;
        hiddeenLayer.activationFunction = activationFunction;
        hiddeenLayer.nOut = nOut;
        hiddeenLayer.weightInit = weightInit;
        hiddeenLayer.nIn = nIn;
        hiddeenLayer.rng = rng;
        return hiddeenLayer;
    }

    @Override
    public HiddenLayer transpose() {
        HiddenLayer hiddeenLayer = new HiddenLayer();
        hiddeenLayer.b = b.dup();
        hiddeenLayer.W = W.transpose();
        if(input != null)
            hiddeenLayer.input = input.transpose();
        if(dist != null)
            hiddeenLayer.dist = dist;
        hiddeenLayer.activationFunction = activationFunction;
        hiddeenLayer.nOut = nIn;
        hiddeenLayer.weightInit = weightInit;
        hiddeenLayer.nIn = nOut;
        hiddeenLayer.concatBiases = concatBiases;
        hiddeenLayer.rng = rng;
        return hiddeenLayer;
    }



    @Override
    public  INDArray activate() {
        INDArray activation =  getActivationFunction().apply(getInput().mmul(getW()).addRowVector(getB()));
        return activation;
    }

    @Override
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