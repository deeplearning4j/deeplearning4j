package org.deeplearning4j.nn.api;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.util.Map;

/**
 *
 * @author Alex Black
 * @author raver119@gmail.com
 */
public interface NeuralNetworkPrototype {
    /*
        Model params section
    */
    INDArray getParams();

    Updater getUpdater();

    double getScore();

    <T> T getConfiguration();

    /*
        Layers section
    */
    // however, we can replicate to actual structure
    Layer[] getLayers();


    /*
        Fitting section
    */
    // we should have unified dataset here
    void fit(DataSet dataSet);

    // should be unified iterator too
    void fit(DataSetIterator iterator);

    // same, iterator unification would be nice to see here
    void pretrain(DataSetIterator iterator);


    /*
        Output section
    */
    Map<String, INDArray> activations(INDArray input);

    INDArray output(INDArray input);

    INDArray[] output(INDArray... input);


    /*
        RNN section
    */
    void rnnClearPreviousState();

    Map<String, Map<String, INDArray>> rnnGetPreviousStates();

    void rnnTimeStep(INDArray... input);


    /*
        Evaluation section
    */
    // why exactly we have Evaluation class AND evaluation code in MLN/CG at the same time?
}
