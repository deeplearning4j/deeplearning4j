package org.deeplearning4j.rl4j.mdp;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.sync.Transition;
import org.deeplearning4j.rl4j.mdp.IMDPRunner;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.policy.Policy;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.util.ArrayUtil;

public abstract class BaseMDPRunner<O extends Encodable, A> implements IMDPRunner<O, A> {

    @Getter @Setter(AccessLevel.PUBLIC) // FIXME: Setter to protected
    INDArray[] history;

    @Getter @Setter
    double maxQ = Double.NaN;

    public void onPreEpoch() {
        history = null;
    }

    @Override
    public A getNextAction(IDQN currentDQN, Policy<O, A> policy, INDArray input) {
        INDArray hstack = getHStack(input);

        INDArray qs = currentDQN.output(hstack);
        int maxAction = Learning.getMaxAction(qs);

        setMaxQ(qs.getDouble(maxAction));
        return policy.nextAction(hstack);
    }
}
