package org.deeplearning4j.nn.graph.multioutput.testlayers;

import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class SplitOutputLayer extends OutputLayer {

    protected Activations twoLabels;

    public SplitOutputLayer(SplitOutputLayerConf conf) {
        super(conf);
    }

    @Override
    public void setLabels(INDArray labels, INDArray labelMask){
        throw new UnsupportedOperationException("Cannot use this method with multiple outputs");
    }

    @Override
    public void setLabels(Activations labels){
        this.twoLabels = labels;
    }

    @Override
    public int numOutputs(){
        return 2;
    }

    @Override
    public Gradients backpropGradient(Gradients epsilons) {

        this.labels = Nd4j.hstack(twoLabels.getAsArray());

        return super.backpropGradient(epsilons);
    }

    @Override
    public Activations activate(boolean training){
        Activations a = super.activate(training);

        INDArray origOut = a.get(0);
        int s1 = origOut.size(1);
        INDArray out1 = origOut.get(NDArrayIndex.all(), NDArrayIndex.interval(0, s1/2));
        INDArray out2 = origOut.get(NDArrayIndex.all(), NDArrayIndex.interval(s1/2, s1));

        return ActivationsFactory.getInstance().createPair(out1, out2);
    }

    @Override
    public double computeScore(Activations layerOutput, Activations labels, double fullNetworkL1, double fullNetworkL2, boolean training) {
        layerOutput = ActivationsFactory.getInstance().create( Nd4j.hstack(layerOutput.getAsArray()));
        labels = ActivationsFactory.getInstance().create(Nd4j.hstack(labels.getAsArray()));
        return super.computeScore(layerOutput, labels, fullNetworkL1, fullNetworkL2, training);
    }
}
