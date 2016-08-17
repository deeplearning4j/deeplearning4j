package org.deeplearning4j.rl4j.learning.sync;

import lombok.Value;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/12/16.
 */
@Value
public class Transition<A> {

    INDArray[] observation;
    A action;
    double reward;
    boolean isTerminal;
    INDArray nextObservation;

    public static INDArray concat(INDArray[] history){
        INDArray arr = Nd4j.concat(0, history);
        if (arr.shape().length > 2)
            arr.muli(1/256f);
        return arr;
    }

    public Transition<A> dup(){
        INDArray[] dupObservation = dup(observation);
        INDArray nextObs = nextObservation.dup();

        return new Transition<>(dupObservation, action, reward, isTerminal, nextObs);
    }

    public static INDArray[] dup(INDArray[] history){
        INDArray[] dupHistory = new INDArray[history.length];
        for (int i = 0; i < history.length; i++) {
            dupHistory[i] = history[i].dup();
        }
        return dupHistory;
    }

    public static INDArray[] append(INDArray[] history, INDArray append){
        INDArray[] appended = new INDArray[history.length];
        appended[0] = append;
        for (int i = 0; i < history.length-1; i++) {
            appended[i+1] = history[i];
        }
        return appended;
    }

}
