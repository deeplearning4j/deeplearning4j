package org.deeplearning4j.nn.layers;


import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.optimize.Solver;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Baseline class for any Neural Network used
 * as a layer in a deep network *
 * @author Adam Gibson
 *
 */
public abstract class BasePretrainNetwork extends BaseLayer {




    private static final long serialVersionUID = -7074102204433996574L;

    protected INDArray doMask;
    private static Logger log = LoggerFactory.getLogger(BasePretrainNetwork.class);

    public BasePretrainNetwork(NeuralNetConfiguration conf) {
        super(conf);
    }

    public BasePretrainNetwork(NeuralNetConfiguration conf, INDArray input) {
        super(conf, input);
    }




    /**
     * Applies sparsity to the passed in hbias gradient
     * @param hBiasGradient the hbias gradient to apply to
     */
    protected void applySparsity(INDArray hBiasGradient) {
        INDArray change = hBiasGradient.mul(conf.getSparsity()).mul(-conf.getLr() * conf.getSparsity());
        hBiasGradient.addi(change);


    }



    @Override
    public double score() {
        if(conf.getLossFunction() != LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
            return  -LossFunctions.score(
                    input,
                    conf.getLossFunction(),
                    transform(input),
                    conf.getL2(),
                    conf.isUseRegularization());
        else {
            return -LossFunctions.reconEntropy(
                    input,
                    getParam(PretrainParamInitializer.BIAS_KEY),
                    getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY),
                    getParam(PretrainParamInitializer.WEIGHT_KEY),
                    conf.getActivationFunction());
        }
    }


    @Override
    public void update(Gradient gradient) {
        setParams(params().addi(gradient.gradient()));
    }

    /**
     * iterate one iteration of the network
     *
     * @param input  the input to iterate on
     */
    @Override
    public void iterate(INDArray input) {
        this.input = input;
        Gradient gradient = getGradient();
        update(gradient);
    }


    protected Gradient createGradient(INDArray wGradient,INDArray vBiasGradient,INDArray hBiasGradient) {
        Gradient ret = new DefaultGradient();
        ret.gradientLookupTable().put(PretrainParamInitializer.VISIBLE_BIAS_KEY,vBiasGradient);
        ret.gradientLookupTable().put(PretrainParamInitializer.BIAS_KEY,hBiasGradient);
        ret.gradientLookupTable().put(PretrainParamInitializer.WEIGHT_KEY,wGradient);
        return ret;
    }


    protected void applyDropOutIfNecessary(INDArray input) {
        if(conf.getDropOut() > 0) {
            this.doMask = Nd4j.rand(input.rows(), input.columns()).gt(conf.getDropOut());
        }

        else
            this.doMask = Nd4j.ones(input.rows(),input.columns());

        //actually apply drop out
        input.muli(doMask);

    }

    @Override
    public void fit() {
        Solver solver = new Solver.Builder()
                .model(this).configure(conf()).listeners(conf.getListeners())
                .build();
        solver.optimize();

    }

    //align input so it can be used in training
    protected INDArray preProcessInput(INDArray input) {
        if(conf.isConcatBiases())
            return Nd4j.hstack(input,Nd4j.ones(input.rows(),1));
        return input;
    }

    public abstract Pair<INDArray,INDArray> sampleHiddenGivenVisible(INDArray v);

    public abstract Pair<INDArray,INDArray> sampleVisibleGivenHidden(INDArray h);





}
