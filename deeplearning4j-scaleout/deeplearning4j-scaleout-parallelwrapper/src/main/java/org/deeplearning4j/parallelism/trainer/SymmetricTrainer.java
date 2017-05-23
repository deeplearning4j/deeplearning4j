package org.deeplearning4j.parallelism.trainer;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.GradientsProcessor;
import org.deeplearning4j.optimize.listeners.SharedGradient;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This trainer implementation does parallel training via gradients broadcasts.
 * After each iteration, gradients from this trainer will be propagated & applied to all other trainers
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class SymmetricTrainer extends DefaultTrainer implements CommunicativeTrainer {
    @Builder.Default protected GradientsProcessor extractor = new GradientsProcessor();

    public SymmetricTrainer(@NonNull Model originalModel, int threadIdx, @NonNull WorkspaceMode mode, @NonNull ParallelWrapper wrapper) {
        super();
        this.originalModel = originalModel;
        this.threadId = threadIdx;
        this.workspaceMode = mode;
        this.parallelWrapper = wrapper;
    }

    public void enqueueGradient(SharedGradient gradient) {
        //log.info("Gradient attached: {}", gradient.getGradient().isAttached());
        extractor.enqueueGradient(gradient);
    }


    @Override
    public boolean averagingRequired() {
        return false;
    }

    @Override
    protected void fit(DataSet dataSet) {
        super.fit(dataSet);

        // gradients should be extracted here
        // and broadcasted to all trainers

        while (!extractor.getOwnGradients().isEmpty()) {
            // TODO: ensure gradients array is detached!!!

            parallelWrapper.broadcastGradients(extractor.getOwnGradients().poll());
        }
    }

    @Override
    protected void fit(MultiDataSet dataSet) {
        super.fit(dataSet);

        // gradients should be extracted here
    }

    @Override
    protected void postInit() {
        super.postInit();

        if (extractor == null)
            extractor = new GradientsProcessor();

        replicatedModel.addListener(extractor);
    }




}
