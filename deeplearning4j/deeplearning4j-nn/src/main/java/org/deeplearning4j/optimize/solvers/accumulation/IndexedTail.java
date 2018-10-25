package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import lombok.var;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.ThresholdCompression;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

@Slf4j
public class IndexedTail {
    // here we store positions of individual consumers
    protected ConcurrentHashMap<Long, AtomicLong> positions = new ConcurrentHashMap<>();

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
        updates.put(idx, update);
    }

    /**
     *
     * @return
     */
    public boolean hasAynthing() {
        val threadId = Thread.currentThread().getId();
        var threadPosition = positions.get(threadId);

        // will be instantiated on first call from any given thread
        if (threadPosition == null) {
            threadPosition = new AtomicLong(0);
            positions.put(threadId, threadPosition);
        }


        return threadPosition.get() < updatesCounter.get();
    }

    public boolean drainTo(@NonNull INDArray array) {
        val threadId = Thread.currentThread().getId();
        var threadPosition = positions.get(threadId);

        // will be instantiated on first call from any given thread
        if (threadPosition == null) {
            threadPosition = new AtomicLong(0);
            positions.put(threadId, threadPosition);
        }

        // we're finding out, how many arrays we should provide
        val delta =  updatesCounter.get() - threadPosition.get();

        for (long e = threadPosition.get(); e < threadPosition.get() + delta; e++) {
            val update = updates.get(e);

            smartDecompress(update, array);
        }

        threadPosition.addAndGet(delta);

        return delta > 0;
    }

    protected INDArray smartDecompress(INDArray encoded, @NonNull INDArray target) {
        INDArray result = target;

        if (encoded.isCompressed() || encoded.data().dataType() == DataBuffer.Type.INT) {
            int encoding = encoded.data().getInt(3);
            if (encoding == ThresholdCompression.FLEXIBLE_ENCODING) {
                Nd4j.getExecutioner().thresholdDecode(encoded, result);
            } else if (encoding == ThresholdCompression.BITMAP_ENCODING) {
                Nd4j.getExecutioner().bitmapDecode(encoded, result);
            } else
                throw new ND4JIllegalStateException("Unknown encoding mode: [" + encoding + "]");
        } else {
            result.addi(encoded);
        }

        return result;
    }
}
