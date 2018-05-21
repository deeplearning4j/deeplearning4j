package org.deeplearning4j.rl4j.learning.async;

import org.deeplearning4j.rl4j.learning.ILearning;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/23/16.
 *
 * Interface configuration for all training method that inherit
 * from AsyncLearning
 */
public interface AsyncConfiguration extends ILearning.LConfiguration {

    int getSeed();

    int getMaxEpochStep();

    int getMaxStep();

    int getNumThread();

    int getNstep();

    int getTargetDqnUpdateFreq();

    int getUpdateStart();

    double getRewardFactor();

    double getGamma();

    double getErrorClamp();

}
