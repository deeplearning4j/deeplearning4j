package org.deeplearning4j.optimize;

import java.io.Serializable;
import java.util.List;

import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.gradient.OutputLayerGradient;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import cc.mallet.optimize.Optimizable;



/**
 * Optimizes the logistic layer for finetuning
 * a multi layer network. This is meant to be used
 * after pretraining.
 * @author Adam Gibson
 *
 */
public class MultiLayerNetworkOptimizer implements Optimizable.ByGradientValue,Serializable,OptimizableByGradientValueMatrix {

    private static final long serialVersionUID = -3012638773299331828L;

    protected BaseMultiLayerNetwork network;

    private static Logger log = LoggerFactory.getLogger(MultiLayerNetworkOptimizer.class);
    private double lr;

    public MultiLayerNetworkOptimizer(BaseMultiLayerNetwork network,double lr) {
        this.network = network;
        this.lr = lr;
    }


    public void optimize(DoubleMatrix labels,double lr,int epochs,TrainingEvaluator eval) {
        network.getOutputLayer().setLabels(labels);

        if(!network.isForceNumEpochs()) {
            network.getOutputLayer().trainTillConvergence(labels,lr,epochs,eval);

            if(network.isShouldBackProp())
                network.backProp(lr, epochs,eval);

        }

        else {
            log.info("Training for " + epochs + " epochs");
            List<DoubleMatrix> activations = network.feedForward();
            DoubleMatrix train = activations.get(activations.size() - 1);

            for(int i = 0; i < epochs; i++) {
                network.getOutputLayer().train(train, labels,lr);
            }


            if(network.isShouldBackProp())
                network.backProp(lr, epochs,eval);

        }



    }

    /**
     *
     * @param labels
     * @param lr
     * @param epochs
     */
    public void optimize(DoubleMatrix labels,double lr,int epochs) {
        network.getOutputLayer().setLabels(labels);
        if(!network.isForceNumEpochs()) {
            network.getOutputLayer().trainTillConvergence(labels,lr,epochs,null);

            if(network.isShouldBackProp())
                network.backProp(lr, epochs);

        }

        else {
            log.info("Training for " + epochs + " epochs");
            for(int i = 0; i < epochs; i++) {
                network.getOutputLayer().train(lr);
                log.info("Error on epoch " + i + " is  " + network.getOutputLayer().score());
            }


            if(network.isShouldBackProp())
                network.backProp(lr, epochs);

        }



    }








    @Override
    public int getNumParameters() {
        return network.getOutputLayer().getW().length + network.getOutputLayer().getB().length;
    }



    @Override
    public void getParameters(double[] buffer) {
        int idx = 0;
        for(int i = 0; i < network.getOutputLayer().getW().length; i++) {
            buffer[idx++] = network.getOutputLayer().getW().get(i);

        }
        for(int i = 0; i < network.getOutputLayer().getB().length; i++) {
            buffer[idx++] = network.getOutputLayer().getB().get(i);
        }
    }



    @Override
    public double getParameter(int index) {
        if(index >= network.getOutputLayer().getW().length) {
            int i = index - network.getOutputLayer().getB().length;
            return network.getOutputLayer().getB().get(i);
        }
        else
            return network.getOutputLayer().getW().get(index);
    }



    @Override
    public void setParameters(double[] params) {
        int idx = 0;
        for(int i = 0; i < network.getOutputLayer().getW().length; i++) {
            network.getOutputLayer().getW().put(i,params[idx++]);
        }


        for(int i = 0; i < network.getOutputLayer().getB().length; i++) {
            network.getOutputLayer().getB().put(i,params[idx++]);
        }
    }



    @Override
    public void setParameter(int index, double value) {
        if(index >= network.getOutputLayer().getW().length) {
            int i = index - network.getOutputLayer().getB().length;
            network.getOutputLayer().getB().put(i,value);
        }
        else
            network.getOutputLayer().getW().put(index,value);
    }



    @Override
    public void getValueGradient(double[] buffer) {
        OutputLayerGradient gradient = network.getOutputLayer().getGradient(lr);

        DoubleMatrix weightGradient = gradient.getwGradient();
        DoubleMatrix biasGradient = gradient.getbGradient();

        int idx = 0;

        for(int i = 0; i < weightGradient.length; i++)
            buffer[idx++] = weightGradient.get(i);
        for(int i = 0; i < biasGradient.length; i++)
            buffer[idx++] = biasGradient.get(i);

    }



    @Override
    public double getValue() {
        return network.score();
    }



    @Override
    public DoubleMatrix getParameters() {
        double[] d = new double[getNumParameters()];
        this.getParameters(d);
        return new DoubleMatrix(d);
    }



    @Override
    public void setParameters(DoubleMatrix params) {
        this.setParameters(params.toArray());
    }



    @Override
    public DoubleMatrix getValueGradient(int iteration) {
        double[] buffer = new double[getNumParameters()];
        getValueGradient(buffer);
        return new DoubleMatrix(buffer);
    }


}
