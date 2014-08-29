package org.deeplearning4j.nn.layers;



import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.linalg.ops.transforms.Transforms;

import org.deeplearning4j.nn.WeightInit;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

/**
 * A layer with a bias and activation function
 * @author Adam Gibson
 */
public abstract class BaseLayer implements Layer {

    protected INDArray W;
    protected INDArray b;
    protected INDArray input;
    protected NeuralNetConfiguration conf;
    protected INDArray dropoutMask;



    public BaseLayer(NeuralNetConfiguration conf,INDArray W, INDArray b, INDArray input) {
        this.input = input;

        if(W == null) {

            this.W = NDArrays.zeros(conf.getnIn(), conf.getnOut());

            for(int i = 0; i < this.W.rows(); i++)
                this.W.putRow(i,NDArrays.create(conf.getDist().sample(this.W.columns())));
        }

        else
            this.W = W;


        if(b == null)
            this.b = NDArrays.zeros(conf.getnOut());
        else
            this.b = b;
        this.conf = conf;
    }



    /**
     * Classify input
     * @param x the input (can either be a matrix or vector)
     * If it's a matrix, each row is considered an example
     * and associated rows are classified accordingly.
     * Each row will be the likelihood of a label given that example
     * @return a probability distribution for each row
     */
    @Override
    public  INDArray preOutput(INDArray x) {
        if(x == null)
            throw new IllegalArgumentException("No null input allowed");

        this.input = x;

        INDArray ret = this.input.mmul(W);
        if(conf.isConcatBiases())
            ret = NDArrays.concatHorizontally(ret,b);
        else
            ret.addiRowVector(b);
        return ret;


    }


    @Override
    public  INDArray activate() {
        INDArray activation =  conf.getActivationFunction().apply(getInput().mmul(getW()).addRowVector(getB()));
        return activation;
    }

    @Override
    public  INDArray activate(INDArray input) {
        if(input != null)
            this.input = Transforms.stabilize(input.dup(), 1);
        return activate();
    }




    @Override
    public NeuralNetConfiguration conf() {
        return conf;
    }

    @Override
    public void setConfiguration(NeuralNetConfiguration conf) {
        this.conf = conf;
    }

    @Override
    public INDArray getW() {
        return W;
    }

    @Override
    public void setW(INDArray W) {
        this.W = W;
    }

    @Override
    public INDArray getB() {
        return b;
    }

    @Override
    public void setB(INDArray b) {
        this.b = b;
    }

    @Override
    public INDArray getInput() {
        return input;
    }

    @Override
    public void setInput(INDArray input) {
        this.input = input;
    }




    protected void applyDropOutIfNecessary(INDArray input) {
        if(conf.getDropOut() > 0) {
            this.dropoutMask = NDArrays.rand(input.rows(), conf.getnOut()).gt(conf.getDropOut());
        }

        else
            this.dropoutMask = NDArrays.ones(input.rows(), conf.getnOut());

        //actually apply drop out
        input.muli(dropoutMask);

    }

    /**
     * Averages the given logistic regression
     * from a mini batch in to this one
     * @param l the logistic regression to average in to this one
     * @param batchSize  the batch size
     */
    public void merge(Layer l,int batchSize) {
        if(conf.isUseRegularization()) {

            W.addi(l.getW().subi(W).div(batchSize));
            b.addi(l.getB().subi(b).div(batchSize));
        }

        else {
            W.addi(l.getW().subi(W));
            b.addi(l.getB().subi(b));
        }

    }


    @Override
    public Layer clone() {
        Layer layer = null;
        try {
            layer =  getClass().newInstance();
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        }

        layer.setB(b.dup());
        layer.setW(W.dup());
        if(input != null)
            layer.setInput(input.dup());
        return layer;

    }

    @Override
    public Layer transpose() {
        Layer layer = null;
        try {
            layer = getClass().newInstance();
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        }
        layer.setB(b.dup());
        layer.setW(W.transpose().dup());
        if(input != null)
            layer.setInput(input.transpose().dup());
        return layer;
    }

    @Override
    public String toString() {
        return "BaseLayer{" +
                "W=" + W +
                ", b=" + b +
                ", input=" + input +
                ", conf=" + conf +
                ", dropoutMask=" + dropoutMask +
                '}';
    }
}
