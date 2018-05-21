package org.deeplearning4j.rl4j.learning.sync;

import lombok.Value;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/12/16.
 *
 * A transition is a SARS tuple
 * State, Action, Reward, (isTerminal), State
 */
@Value
public class Transition<A> {

    INDArray[] observation;
    A action;
    double reward;
    boolean isTerminal;
    INDArray nextObservation;

    /**
     * concat an array history into a single INDArry of as many channel
     * as element in the history array
     * @param history the history to concat
     * @return the multi-channel INDArray
     */
    public static INDArray concat(INDArray[] history) {
        INDArray arr = Nd4j.concat(0, history);
        return arr;
    }

    /**
     * Duplicate this transition
     * @return this transition duplicated
     */
    public Transition<A> dup() {
        INDArray[] dupObservation = dup(observation);
        INDArray nextObs = nextObservation.dup();

        return new Transition<>(dupObservation, action, reward, isTerminal, nextObs);
    }

    /**
     * Duplicate an history
     * @param history the history to duplicate
     * @return a duplicate of the history
     */
    public static INDArray[] dup(INDArray[] history) {
        INDArray[] dupHistory = new INDArray[history.length];
        for (int i = 0; i < history.length; i++) {
            dupHistory[i] = history[i].dup();
        }
        return dupHistory;
    }

    /**
     * append a pixel frame to an history (throwing the last frame)
     * @param history the history on which to append
     * @param append the pixel frame to append
     * @return the appended history
     */
    public static INDArray[] append(INDArray[] history, INDArray append) {
        INDArray[] appended = new INDArray[history.length];
        appended[0] = append;
        for (int i = 0; i < history.length - 1; i++) {
            appended[i + 1] = history[i];
        }
        return appended;
    }

}
