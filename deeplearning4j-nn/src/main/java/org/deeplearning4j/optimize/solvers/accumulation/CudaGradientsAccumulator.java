package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.optimize.api.StepFunction;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

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
    protected List<MemoryWorkspace> workspaces = new ArrayList<>();
    protected List<ReentrantLock> locks = new ArrayList<>();

    protected AtomicInteger workersCounter = new AtomicInteger(0);
    protected ThreadLocal<Integer> index = new ThreadLocal<>();
    protected long initialMemory = 100 * 1024 * 1024L;
    protected int queueSize = 5;
    protected double boundary = 1.0;

    protected CyclicBarrier barrier;

    public CudaGradientsAccumulator(double parties) {
        this(Nd4j.getAffinityManager().getNumberOfDevices(), 1e-3);
    }

    // TODO: delete this one maybe?
    public CudaGradientsAccumulator(int parties) {
        this(parties, 1e-3);
    }

    public CudaGradientsAccumulator(int parties, double threshold) {
        this(parties, new EncodingHandler(threshold), 100 * 1024 * 1024L, 10, 1.0);
    }

    protected CudaGradientsAccumulator(int parties, @NonNull MessageHandler handler, long initialMemory, int queueSize, double boundary) {
        this.parties = parties;
        this.handler = handler;
        this.initialMemory = initialMemory;
        this.queueSize = queueSize;
        this.boundary = boundary;

        // maybe not the best idea in the world, but we'll use cyclic workspace of 25MB to receive updates
        WorkspaceConfiguration configuration = WorkspaceConfiguration.builder()
                .initialSize(initialMemory)
                .policyReset(ResetPolicy.ENDOFBUFFER_REACHED)
                .policyAllocation(AllocationPolicy.STRICT)
                .policySpill(SpillPolicy.FAIL)
                .policyLearning(LearningPolicy.NONE)
                .build();

        if (parties > Nd4j.getAffinityManager().getNumberOfDevices())
            throw new ND4JIllegalStateException("Number of parties ["+ parties +"] should be less or equal to number of devices ["+Nd4j.getAffinityManager().getNumberOfDevices()+"]");

        // pre-create Queues for local workers
        int curDev = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        for (int i = 0; i < parties; i++) {
            messages.add(new LinkedBlockingQueue<INDArray>(queueSize));

            Nd4j.getAffinityManager().unsafeSetDevice(i);
            MemoryWorkspace ws = Nd4j.getWorkspaceManager().createNewWorkspace(configuration,"CGA-" + i, i);
            //ws.enableDebug(true);
            workspaces.add(ws);

            locks.add(new ReentrantLock());
        }
        Nd4j.getAffinityManager().unsafeSetDevice(curDev);

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
     * This method does initialization of given worker wrt Thread-Device Affinity
     */
    @Override
    public void touch() {
        if (index.get() == null) {
            // set index
            int localIndex = Nd4j.getAffinityManager().getDeviceForCurrentThread();

            index.set(localIndex);
        }
    }

    /**
     * This method accepts updates suitable for StepFunction, and accumulates/propagates it across all workers
     *
     * @param array
     */
    @Override
    public void storeUpdate(INDArray array) {

        if (accumulator.get() == null) {
            // we don't want accumulator to be attached to workspaces
            try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                accumulator.set(Nd4j.create(array.shape(), array.ordering()));
            }
        }

        accumulator.get().addi(array);

        handler.broadcastUpdates(accumulator.get());

        try {
            // FIXME: this thing is needed for last update only
            barrier.await(100, TimeUnit.MILLISECONDS);
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

            //try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {

            // we don't want to have same workspace to be accessible by 2 different threads for now
            locks.get(i).lock();

            try (MemoryWorkspace workspace = workspaces.get(i).notifyScopeEntered()) {
                if (array.data().length() > (initialMemory / queueSize) / Nd4j.sizeOfDataType(array.data().dataType()))
                    throw new ND4JIllegalStateException("Not enough memory to handle update: ["+ array.data().length() * Nd4j.sizeOfDataType(array.data().dataType())+" bytes required]. Please increase memory amount for GradientsAccumulator");

                INDArray compressed = array.unsafeDuplication();
                messages.get(i).add(compressed);
            }

            locks.get(i).unlock();
            //}

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


    public static class Builder {
        protected int parties;
        protected double threshold = 1e-3;
        protected long initialMemory = 100 * 1024 * 1024L;
        protected int queueSize = 5;
        protected MessageHandler handler;
        protected Double boundary = null;

        /**
         * This
         * @param parties
         */
        public Builder(int parties) {
            if (parties < 1)
                throw new DL4JInvalidConfigException("Number of parties for GradientsAccumulation should be positive value");

            this.parties = parties;
        }

        /**
         * This method allows to set encoding threshold for this accumulator instance
         *
         * Default value: 1e-3
         * @param threshold
         * @return
         */
        public Builder encodingThreshold(double threshold) {
            this.threshold = threshold;
            return this;
        }

        /**
         * This method enables optional limit for max number of updates per message
         *
         * Default value: 1.0 (no limit)
         * @param boundary positive value in range 0..1
         * @return
         */
        public Builder updatesBoundary(double boundary) {
            if (boundary >= 1.0)
                return this;

            if (boundary <= 0.0)
                throw new DL4JInvalidConfigException("Boundary should have positive value");

            this.boundary = boundary;
            return this;
        }


        /**
         * This method allows to define buffer memory parameters for this GradientsAccumulator
         *
         * Default values: 100MB initialMemory, 5 queueSize
         * @param initialMemory
         * @param queueSize
         * @return
         */
        public Builder memoryParameters(long initialMemory, int queueSize) {
            this.initialMemory = initialMemory;
            this.queueSize = queueSize;
            return this;
        }

        public CudaGradientsAccumulator build() {
            if (handler == null) {
                if (boundary == null)
                    handler = new EncodingHandler(threshold);
                else
                    handler = new EncodingHandler(threshold, boundary);
            }


            return new CudaGradientsAccumulator(parties, handler, initialMemory, queueSize, boundary);
        }
    }
}
