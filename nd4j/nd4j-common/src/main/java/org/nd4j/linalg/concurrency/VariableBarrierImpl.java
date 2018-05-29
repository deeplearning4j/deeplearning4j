package org.nd4j.linalg.concurrency;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;

@Slf4j
public class VariableBarrierImpl implements VariableBarrier {
    protected AtomicInteger phase = new AtomicInteger(0);
    protected AtomicInteger consumers = new AtomicInteger(0);

    protected AtomicInteger first = new AtomicInteger(0);
    protected AtomicInteger second = new AtomicInteger(0);

    protected final boolean truncatedMode;

    protected int[] plans;
    protected AtomicInteger plansPosition = new AtomicInteger(0);

    public VariableBarrierImpl() {
        this(false);
    }

    public VariableBarrierImpl(boolean truncatedMode) {
        this.truncatedMode = truncatedMode;
    }

    @Override
    public void registerConsumers(int numberOfConsumers) {
        blockMainThread();

        // we dont want to overwrite bypass state
        if (phase.get() == 0) {
            consumers.set(numberOfConsumers);
            phase.set(1);
        }
    }

    public void blockMainThread() {
        while (phase.get() != 0 && phase.get() >= 0)
            LockSupport.parkNanos(500);
    }

    protected void updatePlan(int[] consumerPlans) {
        // it's better to sort once
        Arrays.sort(consumerPlans);
/*
        try {
            Thread.sleep(50);
        } catch (InterruptedException e) {
            throw new RuntimeException();
        }
*/

//        log.info("Flight plan: {}", consumerPlans);

/*
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            throw new RuntimeException();
        }
*/

        this.plans = consumerPlans;
    }

    public void registerConsumers(@NonNull int[] consumerPlans) {
        blockMainThread();

        // update array
        updatePlan(consumerPlans);

        // set value for next iteration
        this.consumers.set(getConsumersForIteration(0));

        this.plansPosition.set(1);

        // switch phase to synchronized
        this.phase.set(1);
    }

    @Override
    public void synchronizedBlock() {
        // waiting till we're on right phase
        while (phase.get() != 1 && phase.get() >= 0)
            LockSupport.parkNanos(500);

        // last thread updates variable IF we're no on bypass mode
        if (first.incrementAndGet() == consumers.get() && phase.get() >= 0) {
            second.set(0);
            phase.set(2);
        }
    }

    @Override
    public void desynchronizedBlock() {
        // waiting till we're on right phase
        while (phase.get() != 2 && phase.get() >= 0)
            LockSupport.parkNanos(500);

        //log.info("Iteration: {}; consumers: {}", this.plansPosition.get(), this.consumers.get());

        // last thread sets phase to 0 if we're NOT on bypass mode
        if (second.incrementAndGet() == consumers.get() && phase.get() >= 0) {
            first.set(0);

            // if that's regular symmetric mode - next phase will be registerConsumer
            if (!truncatedMode) {
                phase.set(0);
            }
            // otherwise - next phase will be synchronizedBlock, with variable number of synchonizers
            else {
                val iteration = plansPosition.getAndIncrement();

                val workers = getConsumersForIteration(iteration);
                if (workers > 0) {
                    this.consumers.set(workers);

                    // let's synchronize!
                    phase.set(1);
                } else {
                    this.consumers.set(0);

                    // time to wait
                    phase.set(0);
                }
            }
        }
    }

    /**
     * This method return number of elements in array <= then iteration number provided as argument
     * @param iteration
     * @return
     */
    protected int getConsumersForIteration(int iteration) {
        int cnt = 0;

        // since plans array was sorted in advance, we'll be searching backwards and will do early reset
        for (int e = plans.length - 1; e >= 0; --e) {
            if (plans[e] > iteration)
                cnt++;
            else
                break;
        }

        return cnt;
    }

    @Override
    public void bypassEverything() {
        phase.set(-1);
    }
}
