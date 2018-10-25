package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

@Slf4j
public class IndexedTail {
    // here we store positions of individual consumers
    protected Map<Long, AtomicLong> positions = new ConcurrentHashMap<>();

    // here we store individual updates
    protected Map<Long, INDArray> updates = new ConcurrentHashMap<>();

    protected AtomicLong updatesCounter = new AtomicLong(0);

    protected final int expectedConsumers;

    public IndexedTail(int expectedConsumers) {
        this.expectedConsumers = expectedConsumers;
    }

    /**
     * This mehtod adds update, with optional collapse
     * @param update
     */
    public void put(@NonNull INDArray update) {
        val idx = updatesCounter.getAndIncrement();
    }

    /**
     *
     * @return
     */
    public boolean hasAynthing() {
        return false;
    }

    public INDArray get(Long index) {
        return null;
    }
}
