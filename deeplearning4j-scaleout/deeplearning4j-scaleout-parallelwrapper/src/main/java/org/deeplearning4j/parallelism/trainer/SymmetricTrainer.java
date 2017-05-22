package org.deeplearning4j.parallelism.trainer;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This trainer implementation does parallel training via gradients broadcasts.
 * After each iteration, gradients from this trainer will be propagated & applied to all other trainers
 *
 * @author raver119@gmail.com
 */
@Builder
@Slf4j
@NoArgsConstructor
@AllArgsConstructor
public class SymmetricTrainer extends DefaultTrainer {
    protected INDArray lastGradients;


    @Override
    protected void fit(DataSet dataSet) {
        super.fit(dataSet);

        // gradients should be extracted here
        // and broadcasted to all trainers
        // parallelWrapper.broadcastGradients();
    }

    @Override
    protected void fit(MultiDataSet dataSet) {
        super.fit(dataSet);

        // gradients should be extracted here
    }

    public static class SymmetricTrainerBuilder extends DefaultTrainerBuilder {
        @Override
        public SymmetricTrainerBuilder originalModel(Model originalModel) {
            return (SymmetricTrainerBuilder) super.originalModel(originalModel);
        }

        @Override
        public SymmetricTrainerBuilder replicatedModel(Model replicatedModel) {
            return (SymmetricTrainerBuilder) super.replicatedModel(replicatedModel);
        }

        @Override
        public SymmetricTrainerBuilder queue(LinkedBlockingQueue<DataSet> queue) {
            return (SymmetricTrainerBuilder) super.queue(queue);
        }

        @Override
        public SymmetricTrainerBuilder queueMDS(LinkedBlockingQueue<MultiDataSet> queueMDS) {
            return (SymmetricTrainerBuilder) super.queueMDS(queueMDS);
        }

        @Override
        public SymmetricTrainerBuilder running(AtomicInteger running) {
            return (SymmetricTrainerBuilder) super.running(running);
        }

        @Override
        public SymmetricTrainerBuilder threadId(int threadId) {
            return (SymmetricTrainerBuilder) super.threadId(threadId);
        }

        @Override
        public SymmetricTrainerBuilder shouldUpdate(AtomicBoolean shouldUpdate) {
            return (SymmetricTrainerBuilder) super.shouldUpdate(shouldUpdate);
        }

        @Override
        public SymmetricTrainerBuilder shouldStop(AtomicBoolean shouldStop) {
            return (SymmetricTrainerBuilder) super.shouldStop(shouldStop);
        }

        @Override
        public SymmetricTrainerBuilder thrownException(Exception thrownException) {
            return (SymmetricTrainerBuilder) super.thrownException(thrownException);
        }

        @Override
        public SymmetricTrainerBuilder useMDS(boolean useMDS) {
            return (SymmetricTrainerBuilder) super.useMDS(useMDS);
        }

        @Override
        public SymmetricTrainerBuilder onRootModel(boolean onRootModel) {
            return (SymmetricTrainerBuilder) super.onRootModel(onRootModel);
        }

        @Override
        public SymmetricTrainerBuilder parallelWrapper(ParallelWrapper parallelWrapper) {
            return (SymmetricTrainerBuilder) super.parallelWrapper(parallelWrapper);
        }

        @Override
        public SymmetricTrainerBuilder averagingFrequency(int frequency) {
            return (SymmetricTrainerBuilder) super.averagingFrequency(frequency);
        }
    }
}
