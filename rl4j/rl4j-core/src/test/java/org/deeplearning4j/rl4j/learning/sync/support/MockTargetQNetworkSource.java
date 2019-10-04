package org.deeplearning4j.rl4j.learning.sync.support;

import org.deeplearning4j.rl4j.learning.sync.qlearning.TargetQNetworkSource;
import org.deeplearning4j.rl4j.network.dqn.IDQN;

public class MockTargetQNetworkSource implements TargetQNetworkSource {


    private final IDQN qNetwork;
    private final IDQN targetQNetwork;

    public MockTargetQNetworkSource(IDQN qNetwork, IDQN targetQNetwork) {
        this.qNetwork = qNetwork;
        this.targetQNetwork = targetQNetwork;
    }

    @Override
    public IDQN getTargetQNetwork() {
        return targetQNetwork;
    }

    @Override
    public IDQN getQNetwork() {
        return qNetwork;
    }
}
