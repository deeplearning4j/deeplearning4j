package org.deeplearning4j.rl4j.learning;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.observation.transforms.ObservationTransform;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.observation.Observation;

public class LearningInitializer<O  extends Observation, A, AS extends ActionSpace<A>> implements ILearningInitializer<O, A, AS> {

    @Override
    public Learning.InitMdp<O> initMdp(MDP<O, A, AS> mdp, ObservationTransform transform) {
        Observation obs = mdp.reset();
        if(transform != null) {
            obs = transform.getObservation(obs);
        }
        Observation nextO = obs;

        int step = 0;
        double reward = 0;

        if(transform != null) {
            while (!transform.isReady()) {
                A action = mdp.getActionSpace().noOp(); //by convention should be the NO_OP

                StepReply<O> stepReply = mdp.step(action);
                reward += stepReply.getReward();

                nextO = stepReply.getObservation();
                if(transform != null) {
                    nextO = transform.getObservation(nextO);
                }

                step++;
            }
        }

        return new Learning.InitMdp(step, nextO, reward);
    }
}
