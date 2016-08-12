package org.deeplearning4j.rl4j.learning.async;

import lombok.Getter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.rl4j.network.NeuralNet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/5/16.
 */
public class AsyncGlobal<NN extends NeuralNet> extends Thread {

    final private Logger log = LoggerFactory.getLogger("Global");
    final private NN current;
    final private ConcurrentLinkedQueue<Pair<Gradient, Integer>> queue;
    final private AsyncLearning.AsyncConfiguration a3cc;
    private AtomicInteger T = new AtomicInteger(0);
    @Getter
    private NN target;


    public AsyncGlobal(NN initial, AsyncLearning.AsyncConfiguration a3cc) {
        this.current = initial;
        target = (NN) initial.clone();
        this.a3cc = a3cc;
        queue = new ConcurrentLinkedQueue<>();
    }

    public INDArray[] targetOuput(INDArray batch) {
        synchronized (target) {
            return target.outputAll(batch);
        }
    }

    public boolean isTrainingComplete() {
        return T.get() >= a3cc.getMaxStep();
    }


    public NN cloneCurrent() {
        synchronized (current) {
            return (NN) current.clone();
        }
    }


    public void enqueue(Gradient gradient, Integer nstep) {
        synchronized (this) {
            queue.add(new Pair<>(gradient, nstep));
            notifyAll();
        }
    }

    @Override
    public void run() {
        synchronized (this) {
            while (!isTrainingComplete()) {
                log.info("loop");
                if (!queue.isEmpty()) {
                    log.info("in");
                    Pair<Gradient, Integer> pair = queue.poll();
                    T.addAndGet(pair.getSecond());
                    Gradient gradient = pair.getFirst();
                    current.applyGradient(gradient);
                    log.info("applied gradient");
                    if (T.get() % a3cc.getTargetDqnUpdateFreq() == 0)
                        target = (NN) current.clone();
                } else
                    try {
                        wait();
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
            }
        }

    }
}
