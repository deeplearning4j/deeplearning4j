package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.optimize.api.StepFunction;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This GradientsAccumulator is suited for CUDA backend.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class CudaGradientsAccumulator implements GradientsAccumulator{
    protected ThreadLocal<INDArray> accumulator = new ThreadLocal<>();

    protected int parties;
    protected MessageHandler handler;
    protected List<Queue<INDArray>> messages = new ArrayList<>();

    protected AtomicInteger workersCounter = new AtomicInteger(0);
    protected ThreadLocal<Integer> index = new ThreadLocal<>();

    protected CyclicBarrier barrier;


    public CudaGradientsAccumulator(int parties) {
        this(parties, 1e-3);
    }

    public CudaGradientsAccumulator(int parties, double threshold) {
        this(parties, new EncodingHandler(threshold));
    }

    public CudaGradientsAccumulator(int parties, @NonNull MessageHandler handler) {
        this.parties = parties;
        this.handler = handler;

        // pre-create Queues for local workers
        for (int i = 0; i < parties; i++) {
            messages.add(new LinkedBlockingQueue<INDArray>(128));
        }

        handler.initialize(this);
        barrier = new CyclicBarrier(parties);
    }

    /**
     * This method applies accumulated updates via given StepFunction
     *
     * @param function
     * @param params
     */
    @Override
    public void applyUpdate(StepFunction function, INDArray params, INDArray updates) {
        // nullify given updates first
        updates.assign(0.0f);

        while (!messages.get(index.get()).isEmpty()) {
            INDArray compressed = messages.get(index.get()).poll();

            //log.info("Thread: {}; Compressed: {}", Thread.currentThread().getId(), Arrays.toString(compressed.data().asInt()));

            // FIXME: pass gradients here, and reuse them
            INDArray decoded = Nd4j.getExecutioner().thresholdDecode(compressed, updates);
        }

        function.step(params, updates);
    }

    /**
     * This method applies accumulated updates via given StepFunction
     *
     * @param function
     * @param params
     * @param alpha
     */
    @Override
    public void applyUpdate(StepFunction function, INDArray params, INDArray updates, double alpha) {
        // nullify given updates first
        updates.assign(0.0f);

        while (!messages.get(index.get()).isEmpty()) {
            INDArray compressed = messages.get(index.get()).poll();

            //log.info("Thread: {}; Compressed: {}", Thread.currentThread().getId(), Arrays.toString(compressed.data().asInt()));

            // FIXME: pass gradients here, and reuse them
            INDArray decoded = Nd4j.getExecutioner().thresholdDecode(compressed, updates);
        }

        function.step(params, updates, alpha);
    }

    /**
     * This method accepts updates suitable for StepFunction, and accumulates/propagates it across all workers
     *
     * @param array
     */
    @Override
    public void storeUpdate(INDArray array) {

        if (index.get() == null) {
            try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                accumulator.set(Nd4j.create(array.shape(), array.ordering()));
            }

            index.set(workersCounter.getAndIncrement());
        }

        accumulator.get().addi(array);

        handler.broadcastUpdates(accumulator.get());

        try {
            barrier.await();
        } catch (Exception e) {
            //
        }
    }

    /**
     * This method accepts updates suitable for StepFunction and puts them to the queue, which is used in backpropagation loop
     * <p>
     * PLEASE NOTE: array is expected to be ready for use and match params dimensionality
     *
     * @param array
     */
    @Override
    public void receiveUpdate(INDArray array) {
        // we're replicating COMPRESSED MESSAGES, decompress will be thread-local
        for (int i = 0; i < parties; i++) {
            INDArray compressed = array.unsafeDuplication();
            messages.get(i).add(compressed);

            //log.info("Thread: {}; Copy: {}", Thread.currentThread().getId(), Arrays.toString(compressed.data().asInt()));
        }
    }

    /**
     * This method resets all accumulated updates (if any)
     */
    @Override
    public void reset() {
        // just replace accumulator, gc will do the rest
        accumulator = new ThreadLocal<>();

        // throw away message queues
        for (int i = 0; i < parties; i++) {
            messages.get(i).clear();
        }
    }
}
