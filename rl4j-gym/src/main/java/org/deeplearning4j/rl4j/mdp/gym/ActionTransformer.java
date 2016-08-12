package org.deeplearning4j.rl4j.mdp.gym;


import org.deeplearning4j.rl4j.gym.space.HighLowDiscrete;
import org.deeplearning4j.rl4j.space.DiscreteSpace;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/28/16.
 */
//Convert DQN outputAll to HighLow

public class ActionTransformer extends DiscreteSpace {

    final private int[] availableAction;
    final private HighLowDiscrete hld;

    public ActionTransformer(HighLowDiscrete hld, int[] availableAction) {
        super(availableAction.length);
        this.hld = hld;
        this.availableAction = availableAction;
    }

    @Override
    public Object encode(Integer a) {
        return hld.encode(availableAction[a]);
    }
}
