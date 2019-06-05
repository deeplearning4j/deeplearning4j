package org.deeplearning4j.rl4j.mdp;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.rl4j.learning.IHistoryProcessor;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface IMDPRunner {
    <O extends Encodable, A, AS extends ActionSpace<A>> Learning.InitMdp<O> initMdp(MDP<O, A, AS> mdp);
    INDArray getHStack(INDArray input, IMDPRunner.GetHStackContext context);

    public class GetHStackContext {
        @Getter @Setter
        INDArray[] history;
    }
}
