package org.deeplearning4j.rl4j.support;

import lombok.Getter;
import lombok.Setter;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.learning.async.IAsyncGlobal;
import org.deeplearning4j.rl4j.network.NeuralNet;

import java.util.concurrent.atomic.AtomicInteger;

public class MockAsyncGlobal implements IAsyncGlobal {

    private final NeuralNet current;

    public boolean hasBeenStarted = false;
    public boolean hasBeenTerminated = false;

    public int enqueueCallCount = 0;

    @Setter
    private int maxLoops;
    @Setter
    private int numLoopsStopRunning;
    private int currentLoop = 0;

    public MockAsyncGlobal() {
        this(null);
    }

    public MockAsyncGlobal(NeuralNet current) {
        maxLoops = Integer.MAX_VALUE;
        numLoopsStopRunning = Integer.MAX_VALUE;
        this.current = current;
    }

    @Override
    public boolean isRunning() {
        return currentLoop < numLoopsStopRunning;
    }

    @Override
    public void terminate() {
        hasBeenTerminated = true;
    }

    @Override
    public boolean isTrainingComplete() {
        return ++currentLoop > maxLoops;
    }

    @Override
    public void start() {
        hasBeenStarted = true;
    }

    @Override
    public AtomicInteger getT() {
        return null;
    }

    @Override
    public NeuralNet getCurrent() {
        return current;
    }

    @Override
    public NeuralNet getTarget() {
        return current;
    }

    @Override
    public void enqueue(Gradient[] gradient, Integer nstep) {
        ++enqueueCallCount;
    }
}
