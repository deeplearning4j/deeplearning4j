package org.deeplearning4j.nn;

import static org.deeplearning4j.util.MatrixUtil.stabilizeInput;

import java.io.Serializable;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.nn.activation.Sigmoid;
import org.jblas.DoubleMatrix;


/**
 * Vectorized Hidden Layer
 * @author Adam Gibson
 *
 */
public class HiddenLayer implements Serializable {

    protected static final long serialVersionUID = 915783367350830495L;
    protected int nIn;
    protected int nOut;
    protected DoubleMatrix W;
    protected DoubleMatrix b;
    protected RandomGenerator rng;
    protected DoubleMatrix input;
    protected ActivationFunction activationFunction = new Sigmoid();
    protected RealDistribution dist;
    protected boolean concatBiases = false;
    protected WeightInit weightInit;
    protected HiddenLayer() {}

    public HiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input,ActivationFunction activationFunction) {
        this(nIn,nOut,W,b,rng,input,activationFunction,null);
    }


    public HiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input) {
        this(nIn,nOut,W,b,rng,input,null,null,null);
    }
    public HiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input,WeightInit weightInit) {
        this(nIn,nOut,W,b,rng,input,null,null,weightInit);
    }



    public HiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input,ActivationFunction activationFunction,RealDistribution dist,WeightInit weightInit) {
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

            this.W = DoubleMatrix.zeros(nIn,nOut);

            for(int i = 0; i < this.W.rows; i++)
                this.W.putRow(i,new DoubleMatrix(this.dist.sample(this.W.columns)));
        }

        else
            this.W = W;


        if(b == null)
            this.b = DoubleMatrix.zeros(nOut);
        else
            this.b = b;
    }


    public HiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input,RealDistribution dist,WeightInit weightInit) {
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
                this.W = DoubleMatrix.zeros(nIn,nOut);

                for(int i = 0; i < this.W.rows; i++)
                    this.W.putRow(i,new DoubleMatrix(this.dist.sample(this.W.columns)));

            }
        }

        else
            this.W = W;


        if(b == null)
            this.b = DoubleMatrix.zeros(nOut);
        else
            this.b = b;
    }



    public HiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input,ActivationFunction activationFunction,RealDistribution dist) {
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

            this.W = DoubleMatrix.zeros(nIn,nOut);

            for(int i = 0; i < this.W.rows; i++)
                this.W.putRow(i,new DoubleMatrix(this.dist.sample(this.W.columns)));
        }

        else
            this.W = W;


        if(b == null)
            this.b = DoubleMatrix.zeros(nOut);
        else
            this.b = b;
    }


    public HiddenLayer(int nIn, int nOut, DoubleMatrix W, DoubleMatrix b, RandomGenerator rng,DoubleMatrix input,RealDistribution dist) {
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
                this.W = DoubleMatrix.zeros(nIn,nOut);

                for(int i = 0; i < this.W.rows; i++)
                    this.W.putRow(i,new DoubleMatrix(this.dist.sample(this.W.columns)));

            }
        }

        else
            this.W = W;


        if(b == null)
            this.b = DoubleMatrix.zeros(nOut);
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

    public  DoubleMatrix getW() {
        return W;
    }

    public  void setW(DoubleMatrix w) {
        W = w;
    }

    public  DoubleMatrix getB() {
        return b;
    }

    public  void setB(DoubleMatrix b) {
        this.b = b;
    }

    public  RandomGenerator getRng() {
        return rng;
    }

    public  void setRng(RandomGenerator rng) {
        this.rng = rng;
    }

    public  DoubleMatrix getInput() {
        return input;
    }

    public  void setInput(DoubleMatrix input) {
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
    public  DoubleMatrix activate() {
        DoubleMatrix activation =  getActivationFunction().apply(getInput().mmul(getW()).addRowVector(getB()));
        return activation;
    }

    /**
     * Initialize the layer with the given input
     * and return the activation for this layer
     * given this input
     * @param input the input to use
     * @return
     */
    public  DoubleMatrix activate(DoubleMatrix input) {
        if(input != null)
            this.input = stabilizeInput(input.dup(),1);
        return activate();
    }


    public static class Builder {
        protected int nIn;
        protected int nOut;
        protected DoubleMatrix W;
        protected DoubleMatrix b;
        protected RandomGenerator rng;
        protected DoubleMatrix input;
        protected ActivationFunction activationFunction = new Sigmoid();
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

        public Builder withWeights(DoubleMatrix W) {
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

        public Builder withBias(DoubleMatrix b) {
            this.b = b;
            return this;
        }

        public Builder withInput(DoubleMatrix input) {
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