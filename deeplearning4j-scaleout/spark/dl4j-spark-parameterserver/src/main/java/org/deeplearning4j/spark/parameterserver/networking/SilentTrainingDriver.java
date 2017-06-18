package org.deeplearning4j.spark.parameterserver.networking;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.solvers.accumulation.GradientsAccumulator;
import org.deeplearning4j.spark.parameterserver.networking.messages.SilentUpdatesMessage;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.VoidParameterServer;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.logic.Storage;
import org.nd4j.parameterserver.distributed.logic.completion.Clipboard;
import org.nd4j.parameterserver.distributed.messages.VoidAggregation;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.transport.Transport;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This TrainingDriver implementation is suited ONLY for Spark Master, and handles application & redistribution of incoming encoded messages across distributed network
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class SilentTrainingDriver implements TrainingDriver<SilentUpdatesMessage> {
    protected transient INDArray params;
    protected transient INDArray updates;
    protected transient StepFunction stepFunction;

    protected transient GradientsAccumulator accumulator;

    protected transient VoidConfiguration voidConfiguration;
    protected transient Transport transport;
    protected transient AtomicLong updatesCount;

    /*
        We use this buffer to provide double buffering for incoming messages.
        So we store incoming messages right here, and apply them as time comes
     */
    protected transient BlockingQueue<INDArray> updatesBuffer;

    // these 2 are not used here
    protected transient Storage storage;
    protected transient Clipboard clipboard;


    public SilentTrainingDriver(@NonNull GradientsAccumulator accumulator) {
        log.info("Creating TrainingDriver for worker...");
        this.accumulator = accumulator;
        this.updatesCount = new AtomicLong(0);

        // TODO: make this configurable
        this.updatesBuffer = new LinkedBlockingQueue<>(512);
    }

    public SilentTrainingDriver(@NonNull INDArray params, @NonNull StepFunction stepFunction) {
        log.info("Creating TrainingDriver for master...");
        this.params = params;
        this.stepFunction = stepFunction;
        this.updatesCount = new AtomicLong(0);

        // TODO: make this configurable
        this.updatesBuffer = new LinkedBlockingQueue<>(512);

        // updates are always the same size as params
        try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            this.updates = Nd4j.create(params.shape(), params.ordering());
        }
    }

    @Override
    public void init(@NonNull VoidConfiguration voidConfiguration, @NonNull Transport transport, Storage storage, Clipboard clipboard) {
        this.voidConfiguration = voidConfiguration;
        this.transport = transport;
    }

    @Override
    public void startTraining(SilentUpdatesMessage message) {
        /*
            this method will be invoked on master, and will do 2 things:
            1) silently update params via given StepFunction
            2) propagate this message to everyone
         */
        // if accumulator is defined, we're working at Worker level, so it's not our problem what happens inside
        if (accumulator != null) {
            if (message.getOriginatorId() == transport.getOwnOriginatorId()) {
                return;
            };

            // we can't put more updates here, then (sizeOfQueue - numberOfWorkers)
            try {
                updatesBuffer.put(message.getUpdates());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            accumulator.receiveUpdate(message.getUpdates());
        } else if (params != null && stepFunction != null) {

            // master invokes everything, since that's Silent Worker approach: we want master to be always up-to-date
            synchronized (this) {
                // threshold decoder is inplace & fast
                Nd4j.getExecutioner().thresholdDecode(message.getUpdates(), updates);

                // we apply updates every X iterations
                // TODO: make X configurable :)
                if (updatesCount.incrementAndGet() % 10 == 0) {
                    stepFunction.step(params, updates);

                    // once accumulated updates are applied - reset storage, and wait for other messsages
                    //updates.assign(0.0f);
                    Nd4j.getMemoryManager().memset(updates);
                }
            }

            // we should echo this message to everyone but this shard, but only if there's > 1 shard/client available
            if (transport.numberOfKnownClients() > 1) {
                //log.info("Resending message, skipping {}", message.getOriginatorId());
                transport.sendMessageToAllClients(message, message.getOriginatorId(), transport.getOwnOriginatorId());
            }
        } else
            throw new DL4JInvalidConfigException("Neither GradientsAccumulator or StepFunction is defined!");
    }

    @Override
    public void pickTraining(SilentUpdatesMessage message) {
        // this message won't be ever called
    }

    @Override
    public void aggregationFinished(VoidAggregation aggregation) {
        // no-op
    }

    @Override
    public void finishTraining(long originatorId, long taskId) {
        // no-op
    }

    @Override
    public void addCompletionHook(long originatorId, long frameId, long messageId) {
        // no-op
    }

    @Override
    public String targetMessageClass() {
        return SilentUpdatesMessage.class.getSimpleName();
    }
}
