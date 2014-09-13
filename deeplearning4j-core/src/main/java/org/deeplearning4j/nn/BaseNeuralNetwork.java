package org.deeplearning4j.nn;


import static org.nd4j.linalg.ops.transforms.Transforms.*;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.lang.reflect.Constructor;
import java.util.Arrays;


import org.nd4j.linalg.factory.NDArrayFactory;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.models.classifiers.dbn.DBN;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.api.Persistable;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.nd4j.linalg.learning.AdaGrad;
import org.deeplearning4j.optimize.optimizers.NeuralNetworkOptimizer;
import org.deeplearning4j.plot.NeuralNetPlotter;
import org.deeplearning4j.util.Dl4jReflection;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Baseline class for any Neural Network used
 * as a layer in a deep network such as an {@link DBN}
 * @author Adam Gibson
 *
 */
public abstract class BaseNeuralNetwork implements NeuralNetwork,Persistable {




    private static final long serialVersionUID = -7074102204433996574L;
    /* Weight matrix */
    protected INDArray W;
    /* hidden bias */
    protected INDArray hBias;
    /* visible bias */
    protected INDArray vBias;
    /* input to the network */
    protected INDArray input;
    protected transient NeuralNetworkOptimizer optimizer;
    protected INDArray doMask;
    private static Logger log = LoggerFactory.getLogger(BaseNeuralNetwork.class);
    //previous gradient used for updates
    protected INDArray wGradient,vBiasGradient,hBiasGradient;

    protected int lastMiniBatchSize = 1;

    //adaptive learning rate for each of the biases and weights
    protected AdaGrad wAdaGrad,hBiasAdaGrad,vBiasAdaGrad;

    //configuration of the neural net
    protected NeuralNetConfiguration conf;

    protected BaseNeuralNetwork() {}

    public BaseNeuralNetwork(INDArray input, INDArray W, INDArray hbias, INDArray vbias,NeuralNetConfiguration conf) {
        this.input = input;
        this.W = W;
        this.conf = conf;
        if(this.W != null)
            this.wAdaGrad = new AdaGrad(this.W.rows(),this.W.columns());

        this.vBias = vbias;
        if(this.vBias != null)
            this.vBiasAdaGrad = new AdaGrad(this.vBias.rows(),this.vBias.columns());


        this.hBias = hbias;
        if(this.hBias != null)
            this.hBiasAdaGrad = new AdaGrad(this.hBias.rows(),this.hBias.columns());


        initWeights();

    }

    /**
     * Returns the parameters of the neural network
     *
     * @return the parameters of the neural network
     */
    @Override
    public INDArray params() {
        return Nd4j.toFlattened(W,vBias,hBias);
    }


    public float l2RegularizedCoefficient() {
        return ((float) pow(getW(),2).sum(Integer.MAX_VALUE).element()/ 2.0f)  * conf.getL2() + 1e-6f;
    }

    /**
     * Initialize weights.
     * This includes steps for doing a random initialization of W
     * as well as the vbias and hbias
     */
    protected void initWeights()  {

        if(conf.getnIn() < 1)
            throw new IllegalStateException("Number of visible can not be less than 1");
        if(conf.getnOut() < 1)
            throw new IllegalStateException("Number of hidden can not be less than 1");

        int nVisible = conf.getnIn();
        int nHidden = conf.getnOut();

    	/*
		 * Initialize based on the number of visible units..
		 * The lower bound is called the fan in
		 * The outer bound is called the fan out.
		 * 
		 * Below's advice works for Denoising AutoEncoders and other 
		 * neural networks you will use due to the same baseline guiding principles for
		 * both RBMs and Denoising Autoencoders.
		 * 
		 * Hinton's Guide to practical RBMs:
		 * The weights are typically initialized to small random values chosen from a zero-mean Gaussian with
		 * a standard deviation of about 0.01. Using larger random values can speed the initial learning, but
		 * it may lead to a slightly worse final model. Care should be taken to ensure that the initial weight
		 * values do not allow typical visible vectors to drive the hidden unit probabilities very close to 1 or 0
		 * as this significantly slows the learning.
		 */
        if(this.W == null) {

            this.W = Nd4j.zeros(nVisible,nHidden);

            for(int i = 0; i < this.W.rows(); i++)
                this.W.putRow(i,Nd4j.create(conf.getDist().sample(this.W.columns())));

        }

        this.wAdaGrad = new AdaGrad(this.W.rows(),this.W.columns());

        if(this.hBias == null) {
            this.hBias = Nd4j.zeros(nHidden);
			/*
			 * Encourage sparsity.
			 * See Hinton's Practical guide to RBMs
			 */
            //this.hBias.subi(4);
        }

        this.hBiasAdaGrad = new AdaGrad(hBias.rows(),hBias.columns());


        if(this.vBias == null) {
            if(this.input != null)

                this.vBias = Nd4j.zeros(nVisible);



            else
                this.vBias = Nd4j.zeros(nVisible);
        }

        this.vBiasAdaGrad = new AdaGrad(vBias.rows(),vBias.columns());


    }

    /**
     * The number of parameters for the model
     *
     * @return the number of parameters for the model
     */
    @Override
    public int numParams() {
        return conf.getnIn() * conf.getnOut() + conf.getnIn() + conf.getnOut();
    }

    /**
     * Set the parameters for this model.
     * This expects a linear ndarray which then be unpacked internally
     * relative to the expected ordering of the model
     *
     * @param params the parameters for the model
     */
    @Override
    public void setParams(INDArray params) {
        assert params.length() == numParams() : "Illegal number of parameters passed in, must be of length " + numParams();
        int weightLength = conf.getnIn() * conf.getnOut();
        INDArray weights = params.get(NDArrayIndex.interval(0,weightLength));
        INDArray vBias = params.get(NDArrayIndex.interval(weightLength, weightLength + conf.getnIn()));
        INDArray hBias = params.get(NDArrayIndex.interval(weightLength + conf.getnIn(), weightLength + conf.getnIn() + conf.getnOut()));
        setW(weights.reshape(conf.getnIn(),conf.getnOut()));
        setvBias(vBias.dup());
        sethBias(hBias.dup());

    }

    /**
     * Backprop with the output being the reconstruction
     */
    @Override
    public void backProp(float lr,int iterations,Object[] extraParams) {
        float currRecon = squaredLoss();
        boolean train = true;
        NeuralNetwork revert = clone();
        while(train) {
            if(iterations > iterations)
                break;


            float newRecon = this.squaredLoss();
            //prevent weights from exploding too far in either direction, we want this as close to zero as possible
            if(newRecon > currRecon || currRecon < 0 && newRecon < currRecon) {
                update((BaseNeuralNetwork) revert);
                log.info("Converged for new recon; breaking...");
                break;
            }
            else if(Double.isNaN(newRecon) || Double.isInfinite(newRecon)) {
                update((BaseNeuralNetwork) revert);
                log.info("Converged for new recon; breaking...");
                break;
            }


            else if(newRecon == currRecon)
                break;

            else {
                currRecon = newRecon;
                revert = clone();
                log.info("Recon went down " + currRecon);
            }

            iterations++;

            int plotIterations = conf.getRenderWeightsEveryNumEpochs();
            if(plotIterations > 0) {
                NeuralNetPlotter plotter = new NeuralNetPlotter();
                if(iterations % plotIterations == 0) {
                    plotter.plotNetworkGradient(this,getGradient(extraParams),getInput().rows());
                }
            }

        }

    }

    /**
     * Fit the model to the given data
     *
     * @param data the data to fit the model to
     */
    @Override
    public void fit(INDArray data) {
        fit(data,null);
    }

    /**
     * Applies sparsity to the passed in hbias gradient
     * @param hBiasGradient the hbias gradient to apply to
     */
    protected void applySparsity(INDArray hBiasGradient) {

        if(conf.isUseAdaGrad()) {
            INDArray change = this.hBiasAdaGrad.getLearningRates(hBias).neg().muli(conf.getSparsity()).mul(hBiasGradient.mul(conf.getSparsity()));
            hBiasGradient.addi(change);
        }
        else {
            INDArray change = hBiasGradient.mul(conf.getSparsity()).mul(-conf.getLr() * conf.getSparsity());
            hBiasGradient.addi(change);

        }
    }

    /**
     * Update the gradient according to the configuration such as adagrad, momentum, and sparsity
     * @param gradient the gradient to modify
     * @param iteration the current iteration
     * @param learningRate the learning rate for the current iteration
     */
    protected void updateGradientAccordingToParams(NeuralNetworkGradient gradient,int iteration,float learningRate) {
        INDArray wGradient = gradient.getwGradient();

        INDArray hBiasGradient = gradient.gethBiasGradient();
        INDArray vBiasGradient = gradient.getvBiasGradient();

        //reset adagrad history
        if(iteration != 0 && conf.getResetAdaGradIterations() > 0 &&  iteration % conf.getResetAdaGradIterations() == 0) {
            wAdaGrad.historicalGradient = null;
            hBiasAdaGrad.historicalGradient = null;
            vBiasAdaGrad.historicalGradient = null;
            if(this.W != null && this.wAdaGrad == null)
                this.wAdaGrad = new AdaGrad(this.W.rows(),this.W.columns());

            if(this.vBias != null && this.vBiasAdaGrad == null)
                this.vBiasAdaGrad = new AdaGrad(this.vBias.rows(),this.vBias.columns());


            if(this.hBias != null && this.hBiasAdaGrad == null)
                this.hBiasAdaGrad = new AdaGrad(this.hBias.rows(),this.hBias.columns());

            log.info("Resetting adagrad");
        }

        INDArray wLearningRates = wAdaGrad.getLearningRates(wGradient);
        //change up momentum after so many iterations if specified
        float momentum = conf.getMomentum();
        if(conf.getMomentumAfter() != null && !conf.getMomentumAfter().isEmpty()) {
            int key = conf.getMomentumAfter().keySet().iterator().next();
            if(iteration >= key) {
                momentum = conf.getMomentumAfter().get(key);
            }
        }


        if (conf.isUseAdaGrad())
            wGradient.muli(wLearningRates);
        else
            wGradient.muli(learningRate);

        if (conf.isUseAdaGrad())
            hBiasGradient.muli(hBiasAdaGrad.getLearningRates(hBiasGradient));
        else
            hBiasGradient.muli(learningRate);


        if (conf.isUseAdaGrad())
            vBiasGradient.muli(vBiasAdaGrad.getLearningRates(vBiasGradient));
        else
            vBiasGradient.muli(learningRate);



        //only do this with binary hidden neuralNets
        if (this.hBiasGradient != null && conf.getSparsity() != 0)
            applySparsity(hBiasGradient);


        if (momentum != 0 && this.wGradient != null)
            wGradient.addi(this.wGradient.mul(momentum).addi(wGradient.mul(1 - momentum)));


        if(momentum != 0 && this.vBiasGradient != null)
            vBiasGradient.addi(this.vBiasGradient.mul(momentum).addi(vBiasGradient.mul(1 - momentum)));

        if(momentum != 0 && this.hBiasGradient != null)
            hBiasGradient.addi(this.hBiasGradient.mul(momentum).addi(hBiasGradient.mul(1 - momentum)));




        wGradient.divi(lastMiniBatchSize);
        vBiasGradient.divi(lastMiniBatchSize);
        hBiasGradient.divi(lastMiniBatchSize);


        //simulate post gradient application  and apply the difference to the gradient to decrease the change the gradient has
        if(conf.isUseRegularization() && conf.getL2() > 0) {
            if(conf.isUseAdaGrad())
                wGradient.subi(W.mul(conf.getL2()).muli(wLearningRates));

            else
                wGradient.subi(W.mul(conf.getL2() * learningRate));

        }

        if(conf.isConstrainGradientToUnitNorm()) {
            wGradient.divi(wGradient.norm2(Integer.MAX_VALUE));
            vBiasGradient.divi(vBiasGradient.norm2(Integer.MAX_VALUE));
            hBiasGradient.divi(hBiasGradient.norm2(Integer.MAX_VALUE));
        }


        this.wGradient = wGradient;
        this.vBiasGradient = vBiasGradient;
        this.hBiasGradient = hBiasGradient;

    }


    @Override
    public float score() {
        if(conf.getLossFunction() != LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                return  LossFunctions.score(input,conf.getLossFunction(),transform(input),conf.getL2(),conf.isUseRegularization());
        else {
            return -LossFunctions.reconEntropy(input,hBias,vBias,W,conf.getActivationFunction());
        }
    }

    /**
     * Clears the input from the neural net
     */
    @Override
    public void clearInput() {
        this.input = null;
    }

    @Override
    public AdaGrad getAdaGrad() {
        return this.wAdaGrad;
    }
    @Override
    public void setAdaGrad(AdaGrad adaGrad) {
        this.wAdaGrad = adaGrad;
    }



    @Override
    public NeuralNetwork transpose() {
        try {
            Constructor<?> c =  Dl4jReflection.getEmptyConstructor(getClass());
            c.setAccessible(true);
            NeuralNetwork ret = (NeuralNetwork) c.newInstance();
            ret.setVBiasAdaGrad(hBiasAdaGrad);
            ret.sethBias(vBias.dup());
            ret.setConf(conf);
            ret.setvBias(Nd4j.zeros(hBias.rows(),hBias.columns()));
            ret.setW(W.transpose());
            return ret;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }

    @Override
    public NeuralNetConfiguration conf() {
        return conf;
    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {
        this.conf = conf;
    }

    @Override
    public NeuralNetwork clone() {
        try {
            Constructor<?> c =  Dl4jReflection.getEmptyConstructor(getClass());
            c.setAccessible(true);
            NeuralNetwork ret = (NeuralNetwork) c.newInstance();
            ret.setConf(conf);
            ret.setHbiasAdaGrad(hBiasAdaGrad);
            ret.setVBiasAdaGrad(vBiasAdaGrad);
            ret.sethBias(hBias.dup());
            ret.setvBias(vBias.dup());
            ret.setW(W.dup());
            ret.setAdaGrad(wAdaGrad);
            return ret;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }



    @Override
    public void merge(NeuralNetwork network,int batchSize) {
        W.addi(network.getW().sub(W).divi(batchSize));
        hBias.addi(network.gethBias().sub(hBias).divi(batchSize));
        vBias.addi(network.getvBias().subi(vBias).divi(batchSize));

    }


    /**
     * Copies params from the passed in network
     * to this one
     * @param n the network to copy
     */
    public void update(BaseNeuralNetwork n) {
        this.W = n.W;
        this.conf = n.conf;
        this.hBias = n.hBias;
        this.vBias = n.vBias;
        this.wAdaGrad = n.wAdaGrad;
        this.hBiasAdaGrad = n.hBiasAdaGrad;
        this.vBiasAdaGrad = n.vBiasAdaGrad;
    }

    /**
     * Load (using {@link ObjectInputStream}
     * @param is the input stream to load from (usually a file)
     */
    public void load(InputStream is) {
        try {
            ObjectInputStream ois = new ObjectInputStream(is);
            BaseNeuralNetwork loaded = (BaseNeuralNetwork) ois.readObject();
            update(loaded);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }





    /* (non-Javadoc)
       * @see org.deeplearning4j.nn.api.NeuralNetwork#getW()
       */
    @Override
    public INDArray getW() {
        return W;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.api.NeuralNetwork#setW(org.jblas.INDArray)
     */
    @Override
    public void setW(INDArray w) {
        assert Arrays.equals(w.shape(),new int[]{conf.getnIn(),conf.getnOut()}) : "Invalid shape for w, must be " + Arrays.toString(new int[]{conf.getnIn(),conf.getnOut()});
        W = w;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.api.NeuralNetwork#gethBias()
     */
    @Override
    public INDArray gethBias() {
        return hBias;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.api.NeuralNetwork#sethBias(org.jblas.INDArray)
     */
    @Override
    public void sethBias(INDArray hBias) {
        assert Arrays.equals(hBias.shape(),new int[]{conf.getnOut()}) : "Illegal shape for visible bias, must be of shape " + new int[]{conf.getnOut()};
        this.hBias = hBias;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.api.NeuralNetwork#getvBias()
     */
    @Override
    public INDArray getvBias() {
        return vBias;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.api.NeuralNetwork#setvBias(org.jblas.INDArray)
     */
    @Override
    public void setvBias(INDArray vBias) {
        assert Arrays.equals(vBias.shape(),new int[]{conf.getnIn()}) : "Illegal shape for visible bias, must be of shape " + Arrays.toString(new int[]{conf.getnIn()});
        this.vBias = vBias;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.api.NeuralNetwork#getRng()
     */
    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.api.NeuralNetwork#getInput()
     */
    @Override
    public INDArray getInput() {
        return input;
    }

    /* (non-Javadoc)
     * @see org.deeplearning4j.nn.api.NeuralNetwork#setInput(org.jblas.INDArray)
     */
    @Override
    public void setInput(INDArray input) {
        this.input = input;
    }


    @Override
    public AdaGrad gethBiasAdaGrad() {
        return hBiasAdaGrad;
    }
    @Override
    public void setHbiasAdaGrad(AdaGrad adaGrad) {
        this.hBiasAdaGrad = adaGrad;
    }
    @Override
    public AdaGrad getVBiasAdaGrad() {
        return this.vBiasAdaGrad;
    }
    @Override
    public void setVBiasAdaGrad(AdaGrad adaGrad) {
        this.vBiasAdaGrad = adaGrad;
    }
    /**
     * Write this to an object output stream
     * @param os the output stream to write to
     */
    public void write(OutputStream os) {
        try {
            ObjectOutputStream os2 = new ObjectOutputStream(os);
            os2.writeObject(this);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * All neural networks are based on this idea of
     * minimizing reconstruction error.
     * Both RBMs and Denoising AutoEncoders
     * have a component for reconstructing, ala different implementations.
     *
     * @param x the input to transform
     * @return the reconstructed input
     */
    public abstract INDArray transform(INDArray x);



    protected void applyDropOutIfNecessary(INDArray input) {
        if(conf.getDropOut() > 0) {
            this.doMask = Nd4j.rand(input.rows(), input.columns()).gt(conf.getDropOut());
        }

        else
            this.doMask = Nd4j.ones(input.rows(),input.columns());

        //actually apply drop out
        input.muli(doMask);

    }


    public float squaredLoss() {
        INDArray squaredDiff = pow(transform(input).sub(input),2);
        float loss = (float) squaredDiff.sum(Integer.MAX_VALUE).element() / input.rows();
        if(conf.isUseRegularization()) {
            loss += 0.5 * conf.getL2() * (float) pow(W,2).sum(Integer.MAX_VALUE).element();
        }

        return loss;
    }




    @Override
    public INDArray hBiasMean() {
        INDArray hbiasMean = getInput().mmul(getW()).addRowVector(gethBias());
        return hbiasMean;
    }

    //align input so it can be used in training
    protected INDArray preProcessInput(INDArray input) {
        if(conf.isConcatBiases())
            return Nd4j.hstack(input,Nd4j.ones(input.rows(),1));
        return input;
    }

    @Override
    public void iterationDone(int epoch) {
        int plotEpochs = conf.getRenderWeightsEveryNumEpochs();
        if(plotEpochs <= 0)
            return;
        if(epoch % plotEpochs == 0 || epoch == 0) {
            NeuralNetPlotter plotter = new NeuralNetPlotter();
            plotter.plotNetworkGradient(this,this.getGradient(new Object[]{1,0.001,1000}),getInput().rows());
        }
    }
    public static class Builder<E extends BaseNeuralNetwork> {
        private E ret = null;
        private INDArray W;
        protected Class<? extends NeuralNetwork> clazz;
        private INDArray vBias;
        private INDArray hBias;
        private INDArray input;
        private NeuralNetConfiguration conf;

        public Builder<E> configure(NeuralNetConfiguration conf) {
            this.conf = conf;
            return this;
        }



        @SuppressWarnings("unchecked")
        public E buildEmpty() {
            try {
                return (E) clazz.newInstance();
            } catch (InstantiationException | IllegalAccessException e) {
                throw new RuntimeException(e);
            }
        }



        public Builder<E> withClazz(Class<? extends BaseNeuralNetwork> clazz) {
            this.clazz = clazz;
            return this;
        }


        public Builder<E> withInput(INDArray input) {
            this.input = input;
            return this;
        }

        public Builder<E> asType(Class<E> clazz) {
            this.clazz = clazz;
            return this;
        }


        public Builder<E> withWeights(INDArray W) {
            this.W = W;
            return this;
        }

        public Builder<E> withVisibleBias(INDArray vBias) {
            this.vBias = vBias;
            return this;
        }

        public Builder<E> withHBias(INDArray hBias) {
            this.hBias = hBias;
            return this;
        }




        public E build() {
            return buildWithInput();

        }


        @SuppressWarnings("unchecked")
        private  E buildWithInput()  {
            Constructor<?>[] c = clazz.getDeclaredConstructors();
            for(int i = 0; i < c.length; i++) {
                Constructor<?> curr = c[i];
                curr.setAccessible(true);
                Class<?>[] classes = curr.getParameterTypes();
                //input matrix found
                if(classes != null && classes.length > 0 && classes[0].isAssignableFrom(INDArray.class)) {
                    try {
                        ret = (E) curr.newInstance(input,W, hBias,vBias,conf);
                        return ret;
                    }catch(Exception e) {
                        throw new RuntimeException(e);
                    }

                }
            }
            return ret;
        }


    }



}
