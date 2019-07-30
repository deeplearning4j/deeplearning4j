package org.deeplearning4j.rl4j.support;

import org.deeplearning4j.rl4j.learning.sync.IExpReplay;
import org.deeplearning4j.rl4j.learning.sync.Transition;

import java.util.ArrayList;
import java.util.List;

public class MockExpReplay implements IExpReplay<Integer> {

    public List<Transition<Integer>> transitions = new ArrayList<>();

    @Override
    public ArrayList<Transition<Integer>> getBatch() {
        return null;
    }

    @Override
    public void store(Transition<Integer> transition) {
        transitions.add(transition);
    }
}
