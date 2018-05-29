package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.util.ThreadUtils;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.ThresholdCompression;
import org.nd4j.linalg.concurrency.SynchronizableConsumer;
import org.nd4j.linalg.concurrency.VariableBarrier;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.AtomicThrowable;

import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

/**
 * This GradientsAccumulator is suited for CUDA backend.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class EncodedGradientsAccumulator implements GradientsAccumulator, Registerable, SynchronizableConsumer {
    protected ThreadLocal<INDArray> accumulator = new ThreadLocal<>();

    protected int parties;
    protected MessageHandler handler;
    protected List<BlockingQueue<INDArray>> messages = new ArrayList<>();
    protected List<MemoryWorkspace> workspaces = new ArrayList<>();
    protected List<ReentrantLock> locks = new ArrayList<>();

    protected AtomicInteger workersCounter = new AtomicInteger(0);
    protected ThreadLocal<Integer> index = new ThreadLocal<>();
    protected long initialMemory = 100 * 1024 * 1024L;
    protected int queueSize = 5;
    protected Double boundary = 1.0;

    protected Queue<INDArray> externalSource;

    protected AtomicBoolean isFirst = new AtomicBoolean(false);
    protected AtomicBoolean isDone = new AtomicBoolean(true);

    protected AtomicInteger barrier = new AtomicInteger(0);
    protected AtomicInteger secondary = new AtomicInteger(0);
    protected AtomicBoolean registered = new AtomicBoolean(false);
    protected AtomicBoolean bypassMode = new AtomicBoolean(false);
    protected final AtomicInteger currentConsumers = new AtomicInteger(0);

    protected final AtomicThrowable throwable = new AtomicThrowable();

    protected boolean isDebug = true;
    protected final boolean relocatable;
    protected VariableBarrier realBarrier;

    protected WorkspaceConfiguration appliedConfiguration = WorkspaceConfiguration.builder().minSize(5 * 1024 * 1024L)
                    .overallocationLimit(0.3).policyMirroring(MirroringPolicy.FULL).policySpill(SpillPolicy.REALLOCATE)
                    .policyLearning(LearningPolicy.FIRST_LOOP).policyReset(ResetPolicy.BLOCK_LEFT).build();

    public EncodedGradientsAccumulator(double parties) {
        this(Nd4j.getAffinityManager().getNumberOfDevices(), 1e-3);
    }

    // TODO: delete this one maybe?
    public EncodedGradientsAccumulator(int parties) {
        this(parties, 1e-3);
    }

    public EncodedGradientsAccumulator(int parties, double threshold) {
        this(parties, new EncodingHandler(threshold), 100 * 1024 * 1024L, 10, 1.0);
    }

    protected EncodedGradientsAccumulator(int parties, @NonNull MessageHandler handler, long initialMemory,
                    int queueSize, Double boundary) {
        this.parties = parties;
        this.handler = handler;
        this.initialMemory = initialMemory;
        this.queueSize = queueSize;
        this.boundary = boundary;

        // maybe not the best idea in the world, but we'll use cyclic workspace of 25MB to receive updates
        WorkspaceConfiguration configuration = WorkspaceConfiguration.builder().initialSize(initialMemory)
                        .policyReset(ResetPolicy.ENDOFBUFFER_REACHED).policyAllocation(AllocationPolicy.STRICT)
                        .policySpill(SpillPolicy.FAIL).policyLearning(LearningPolicy.NONE).build();


        // we want to know, if we'll have to relocate data if accessed from different threads/devices
        relocatable = Nd4j.getAffinityManager().getNumberOfDevices() > 1
                        && !Nd4j.getAffinityManager().isCrossDeviceAccessSupported();

        int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();

        // we are going to take single-device systems as edge case: cpu & small models at single-gpu systems.
        if (parties > numDevices && numDevices != 1)
            throw new ND4JIllegalStateException("Number of parties [" + parties
                            + "] should be less or equal to number of devices [" + numDevices + "]");

        // pre-create Queues for local workers
        int curDev = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        for (int i = 0; i < parties; i++) {
            messages.add(new LinkedBlockingQueue<INDArray>(queueSize));

            // we don't want device index to step out of boundaries here
            int cDevice = numDevices > 1 ? i % numDevices : 0;

            Nd4j.getAffinityManager().unsafeSetDevice(cDevice);
            MemoryWorkspace ws = Nd4j.getWorkspaceManager().createNewWorkspace(configuration, "CGA-" + i, cDevice);
            //ws.enableDebug(true);
            workspaces.add(ws);

            locks.add(new ReentrantLock());
        }
        Nd4j.getAffinityManager().unsafeSetDevice(curDev);

        handler.initialize(this);
    }

    /**
     * This method returns optimal bufferSize for a given model
     *
     * We know, that updates are guaranteed to have MAX size of params / 16. So, here we go.
     * I.e. for model with 100m params, that's 400m of floats (or 800m of doubles)
     * The worst case for us is bitmap encoding, that takes 2 bits to encode each gradient value
     *
     * so, for float in worst case we'll have (100m / 16) int elements. So, our buffer size will be 6.25m * queueSize * 4 bytes per int
     *
     * @param paramsLength
     * @param numWorkers
     * @param queueSize
     * @return
     */
    public static long getOptimalBufferSize(long paramsLength, int numWorkers, int queueSize) {
        // we add 64kb just for future proof volatility
        val bufferSize = ((paramsLength / 16) + 65536) * numWorkers * queueSize * 4;
        return bufferSize;
    }

    @Override
    public void setVariableBarrier(@NonNull VariableBarrier barrier) {
        this.realBarrier = barrier;
    }

    @Override
    public VariableBarrier getVariableBarrier() {
        return realBarrier;
    }

    public static long getOptimalBufferSize(Model model, int numWorkers, int queueSize) {
        return getOptimalBufferSize(model.params().length(), numWorkers, queueSize);
    }

    @Override
    public void fallbackToSingleConsumerMode(boolean reallyFallback) {
        if (externalSource != null && externalSource instanceof Registerable)
            ((Registerable) externalSource).fallbackToSingleConsumerMode(reallyFallback);

        bypassMode.set(reallyFallback);
    }

    @Override
    public void registerConsumers(int numConsumers) {
        // we don't want double spending here
        if (registered.get()) {
            if (isDebug)
                log.info("Master thread locks at RC");

            while (registered.get()) {
                ThreadUtils.uncheckedSleep(1);
                if (throwable.isTriggered())
                    throw new RuntimeException(throwable.get());
            }

            if (isDebug)
                log.info("Master thread unlocks at RC");
        }

        // we're passing number of consumers for current session to externalSource, if applicable
        if (externalSource != null && externalSource instanceof Registerable)
            ((Registerable) externalSource).registerConsumers(numConsumers);

        currentConsumers.set(numConsumers);
        registered.set(true);
    }

    protected void synchronize(int consumers) {
        synchronize(consumers, false);
    }

    protected void synchronize(int consumers, boolean finalLock) {
        if (consumers == 1 || bypassMode.get()) {
            if (finalLock)
                registered.set(false);

            return;
        }

        if (isDebug)
            log.info("thread {} locking at CGA: {}", Thread.currentThread().getId(), currentConsumers.get());

        // any first thread entering this block - will reset this field to false
        isDone.compareAndSet(true, false);

        // last thread will set isDone to true
        if (barrier.incrementAndGet() == consumers) {
            secondary.set(0);
            barrier.set(0);
            isFirst.set(false);
            isDone.set(true);
        } else {
            // just wait, till last thread will set isDone to true
            while (!isDone.get()) {
                ThreadUtils.uncheckedSleep(1);
                if (throwable.isTriggered())
                    throw new RuntimeException(throwable.get());
            }
        }

        // second lock here needed only to ensure we won't get overrun over isDone flag
        if (secondary.incrementAndGet() == consumers) {
            if (finalLock)
                registered.set(false);

            isFirst.set(true);
        } else {
            while (!isFirst.get()) {
                ThreadUtils.uncheckedSleep(1);
                if (throwable.isTriggered())
                    throw new RuntimeException(throwable.get());
            }
        }

        if (isDebug)
            log.info("thread {} unlocking at CGA: {}", Thread.currentThread().getId(), currentConsumers.get());

    }

    /**
     * This method applies accumulated updates via given StepFunction
     *
     * @param function
     * @param params
     */
    @Override
    public void applyUpdate(StepFunction function, INDArray params, INDArray updates) {
        try {
            // nullify given updates first
            Nd4j.getMemoryManager().memset(updates);

            int cnt = 0;
            while (!messages.get(index.get()).isEmpty()) {
                INDArray compressed = messages.get(index.get()).poll();

                int encoding = compressed.data().getInt(3);
                if (encoding == ThresholdCompression.FLEXIBLE_ENCODING)
                    Nd4j.getExecutioner().thresholdDecode(compressed, updates);
                else if (encoding == ThresholdCompression.BITMAP_ENCODING)
                    Nd4j.getExecutioner().bitmapDecode(compressed, updates);
                else
                    throw new DL4JInvalidConfigException("Unknown compression header received: " + encoding);

                cnt++;
            }

            if (cnt > 0 && isDebug)
                log.info("Thread {}>Local updates to be applied: {}", Thread.currentThread().getId(), cnt);

            if (externalSource != null) {
                log.info("Thread {} entering applier", Thread.currentThread().getId());

                int ent = 0;
                while (!externalSource.isEmpty()) {
                    INDArray compressed = externalSource.poll();

                    // if we have multiple devices without p2p support - just duplicate messages right from host side
                    if (relocatable) {
                        try (MemoryWorkspace workspace = Nd4j.getWorkspaceManager()
                                        .getAndActivateWorkspace(appliedConfiguration, "CGA_APPLY")) {
                            INDArray compressed_copy = compressed.unsafeDuplication(true);

                            int encoding = compressed.data().getInt(3);
                            if (encoding == ThresholdCompression.FLEXIBLE_ENCODING)
                                Nd4j.getExecutioner().thresholdDecode(compressed_copy, updates);
                            else if (encoding == ThresholdCompression.BITMAP_ENCODING)
                                Nd4j.getExecutioner().bitmapDecode(compressed_copy, updates);
                            else
                                throw new DL4JInvalidConfigException(
                                                "Unknown compression header received: " + encoding);
                        }
                    } else {
                        int encoding = compressed.data().getInt(3);
                        if (encoding == ThresholdCompression.FLEXIBLE_ENCODING)
                            Nd4j.getExecutioner().thresholdDecode(compressed, updates);
                        else if (encoding == ThresholdCompression.BITMAP_ENCODING)
                            Nd4j.getExecutioner().bitmapDecode(compressed, updates);
                        else
                            throw new DL4JInvalidConfigException("Unknown compression header received: " + encoding);
                    }
                    cnt++;
                    ent++;
                }

                if (isDebug)
                    log.info("thread {} finished at Externals", Thread.currentThread().getId());

                if (ent > 0 && isDebug)
                    log.info("External updates to be applied: {}", ent);
            }

            //synchronize(currentConsumers.get(), true);

            if (realBarrier != null)
                realBarrier.desynchronizedBlock();

            // TODO: average updates probably?

            if (cnt > 0)
                function.step(params, updates);
        } catch (Exception e) {
            throwable.setIfFirst(e);
            throw new RuntimeException(e);
        }
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
        try {
            // nullify given updates first
            Nd4j.getMemoryManager().memset(updates);
            //updates.assign(0.0);

            int cnt = 0;
            while (!messages.get(index.get()).isEmpty()) {
                INDArray compressed = messages.get(index.get()).poll();

                int encoding = compressed.data().getInt(3);
                if (encoding == ThresholdCompression.FLEXIBLE_ENCODING)
                    Nd4j.getExecutioner().thresholdDecode(compressed, updates);
                else if (encoding == ThresholdCompression.BITMAP_ENCODING)
                    Nd4j.getExecutioner().bitmapDecode(compressed, updates);
                else
                    throw new DL4JInvalidConfigException("Unknown compression header received: " + encoding);

                cnt++;
            }

            if (cnt > 0 && isDebug)
                log.info("Local updates to be applied: {}", cnt);

            if (externalSource != null) {
                int ent = 0;
                while (!externalSource.isEmpty()) {
                    INDArray compressed = externalSource.poll();


                    // if we have multiple devices without p2p support - just duplicate messages right from host side
                    if (relocatable) {
                        try (MemoryWorkspace workspace = Nd4j.getWorkspaceManager()
                                        .getAndActivateWorkspace(appliedConfiguration, "CGA_APPLY")) {
                            INDArray compressed_copy = compressed.unsafeDuplication(true);
                            int encoding = compressed.data().getInt(3);
                            if (encoding == ThresholdCompression.FLEXIBLE_ENCODING)
                                Nd4j.getExecutioner().thresholdDecode(compressed_copy, updates);
                            else if (encoding == ThresholdCompression.BITMAP_ENCODING)
                                Nd4j.getExecutioner().bitmapDecode(compressed_copy, updates);
                            else
                                throw new DL4JInvalidConfigException(
                                                "Unknown compression header received: " + encoding);
                        }
                    } else {
                        int encoding = compressed.data().getInt(3);
                        if (encoding == ThresholdCompression.FLEXIBLE_ENCODING)
                            Nd4j.getExecutioner().thresholdDecode(compressed, updates);
                        else if (encoding == ThresholdCompression.BITMAP_ENCODING)
                            Nd4j.getExecutioner().bitmapDecode(compressed, updates);
                        else
                            throw new DL4JInvalidConfigException("Unknown compression header received: " + encoding);
                    }
                    cnt++;
                    ent++;
                }

                if (ent > 0 && isDebug)
                    log.info("External updates to be applied: {}", ent);
            }

            //synchronize(currentConsumers.get(), true);

            if (realBarrier != null)
                realBarrier.desynchronizedBlock();

            // TODO: average updates? might have sense

            if (cnt > 0)
                function.step(params, updates, alpha);
        } catch (Exception e) {
            throwable.setIfFirst(e);
            throw new RuntimeException(e);
        }
    }

    /**
     * This method allows to pass external updates to accumulator, they will be populated across all workers using this GradientsAccumulator instance
     *
     * @param source
     */
    @Override
    public void setExternalSource(Queue<INDArray> source) {
        this.externalSource = source;
    }

    /**
     * This method does initialization of given worker wrt Thread-Device Affinity
     */
    @Override
    public void touch() {
        if (index.get() == null) {
            // set index
            int numDevces = Nd4j.getAffinityManager().getNumberOfDevices();

            /*
                if we have > 1 computational device, we assign workers to workspaces "as is", as provided via AffinityManager
             */
            if (numDevces > 1 && parties > 1) {
                int localIndex = Nd4j.getAffinityManager().getDeviceForCurrentThread();

                index.set(localIndex);
            } else {
                // if we have only 1 device (like cpu system, or single gpu), just attach consumer via flat index
                index.set(workersCounter.getAndIncrement());
            }
        }
    }

    /**
     * This method accepts updates suitable for StepFunction, and accumulates/propagates it across all workers
     *
     * @param array
     */
    @Override
    public void storeUpdate(INDArray array) {
        try {
            if (accumulator.get() == null) {
                // we don't want accumulator to be attached to workspaces
                try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    accumulator.set(Nd4j.create(array.shape(), array.ordering()));
                }
            }

            // accumulate gradients updates in residental array
            accumulator.get().addi(array);

            if (isDebug)
                log.info("thread {} locking at Register", Thread.currentThread().getId());

            // block until ParallelWrapper sends us message about number of threads in this cycle
            /*
            if (!bypassMode.get())
                while (!registered.get()) {
                    ThreadUtils.uncheckedSleep(1);
                    if (throwable.isTriggered())
                        throw new RuntimeException(throwable.get());
                }
                */

            if (isDebug)
                log.info("thread {} unlocking at Register", Thread.currentThread().getId());

            // propagate changes & modify accumulator
            handler.broadcastUpdates(accumulator.get());

            // we're blocking here, untill all done broadcasting updates
            //synchronize(currentConsumers.get());

            if (realBarrier != null)
                realBarrier.synchronizedBlock();

        } catch (Exception e) {
            throwable.setIfFirst(e);
            throw new RuntimeException(e);
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
        try {
            // we're replicating COMPRESSED MESSAGES, decompression will be thread-local
            for (int i = 0; i < parties; i++) {
                // we don't want to have same workspace to be accessible by 2 different threads for now
                /*
                    With synchronized external data, it's impossible to deadlock here.
                    Each worker is guaranteed to have at least NUM_WORKERS slots in buffer.
                    So we use this lock just to ensure thread-safety of corresponding workspaces
                */
                locks.get(i).lock();

                try (MemoryWorkspace workspace = workspaces.get(i).notifyScopeEntered()) {
                    // we might just scope out of workspace here, instead of throwing error out
                    if (array.data().length() > (initialMemory / queueSize)
                                    / Nd4j.sizeOfDataType(array.data().dataType()))
                        throw new ND4JIllegalStateException("Not enough memory to handle update: ["
                                        + array.data().length() * Nd4j.sizeOfDataType(array.data().dataType())
                                        + " bytes required]. Please increase memory amount for GradientsAccumulator");

                    INDArray compressed = array.unsafeDuplication();
                    try {
                        messages.get(i).put(compressed);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        log.info("Something bad at index_{}", i);
                        throw new RuntimeException(e);
                    }
                }

                locks.get(i).unlock();
            }
        } catch (Exception e) {
            throwable.setIfFirst(e);
            throw new RuntimeException(e);
        }
    }

    /**
     * This method resets all accumulated updates (if any)
     */
    @Override
    public void reset() {
        // just replace accumulator, gc will do the rest
        accumulator = new ThreadLocal<>();

        // resetting this counter too
        workersCounter.set(0);

        // reset indexes too
        index = new ThreadLocal<>();

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
                throw new DL4JInvalidConfigException(
                                "Number of parties for GradientsAccumulation should be positive value");

            this.parties = parties;
        }

        /**
         * This method allows to specify MessageHandler instance
         *
         * Default value: EncodingHandler
         * @param handler
         * @return
         */
        public Builder messageHandler(@NonNull MessageHandler handler) {
            this.handler = handler;
            return this;
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

        public EncodedGradientsAccumulator build() {
            if (handler == null) {
                if (boundary == null)
                    handler = new EncodingHandler(threshold);
                else
                    handler = new EncodingHandler(threshold, boundary);
            }

            EncodedGradientsAccumulator accumulator =
                            new EncodedGradientsAccumulator(parties, handler, initialMemory, queueSize, boundary);

            return accumulator;
        }
    }
}
