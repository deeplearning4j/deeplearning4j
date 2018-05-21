package org.deeplearning4j.rl4j.learning;

import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/19/16.
 *
 * A common interface that any training method should implement
 */
public interface ILearning<O extends Encodable, A, AS extends ActionSpace<A>> extends StepCountable {

    Policy<O, A> getPolicy();

    void train();

    int getStepCounter();

    LConfiguration getConfiguration();

    MDP<O, A, AS> getMdp();


    interface LConfiguration {

        int getSeed();

        int getMaxEpochStep();

        int getMaxStep();

        double getGamma();
    }

}
