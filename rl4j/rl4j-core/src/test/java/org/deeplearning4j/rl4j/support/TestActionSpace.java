package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.rl4j.space.ActionSpace;

public class TestActionSpace implements ActionSpace<Integer> {
    @Override
    public Integer randomAction() {
        return null;
    }

    @Override
    public void setSeed(int i) {

    }

    @Override
    public Object encode(Integer integer) {
        return null;
    }

    @Override
    public int getSize() {
        return 0;
    }

    @Override
    public Integer noOp() {
        return -1;
    }
}
