package org.deeplearning4j.nn.graph.multioutput.testlayers;

import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.feedforward.dense.DenseLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

public class SplitDenseLayer extends DenseLayer {
    public SplitDenseLayer(SplitDenseLayerConf conf) {
        super(conf);
    }

    @Override
    public int numOutputs(){
        return 2;
    }

    @Override
    public Gradients backpropGradient(Gradients epsilons) {
        INDArray eps1 = epsilons.get(0);
        INDArray eps2 = epsilons.get(1);
        INDArray eps = Nd4j.hstack(eps1, eps2);

        Gradients g = GradientsFactory.getInstance().create(eps, epsilons.getParameterGradients());
        return super.backpropGradient(g);
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
}
