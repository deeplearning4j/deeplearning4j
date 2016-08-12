package org.deeplearning4j.rl4j.learning;

import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.policy.Policy;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/19/16.
 */
//TODO make it an interface and able to customise chart by the user
public interface ILearning<O extends Encodable, A, AS extends ActionSpace<A>> extends StepCountable {

    public abstract Policy<O, A> getPolicy();

    public abstract void train();

    public abstract int getStepCounter();

    public abstract LConfiguration getConfiguration();

    public abstract MDP<O, A, AS> getMdp();


    public interface LConfiguration {
        int getSeed();

        int getMaxEpochStep();

        int getMaxStep();

        double getGamma();
    }

}
