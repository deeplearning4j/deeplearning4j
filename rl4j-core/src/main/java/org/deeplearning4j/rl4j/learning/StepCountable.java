package org.deeplearning4j.rl4j.learning;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 *
 * Express the ability to count the number of step of the current training.
 * Factorisation between thread in async and learning process.
 */
public interface StepCountable {

    public int getStepCounter();
}
