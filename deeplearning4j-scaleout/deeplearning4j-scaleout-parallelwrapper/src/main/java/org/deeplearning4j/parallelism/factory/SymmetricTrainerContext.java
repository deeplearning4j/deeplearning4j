package org.deeplearning4j.parallelism.factory;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.parallelism.MagicQueue;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.parallelism.trainer.DefaultTrainer;
import org.deeplearning4j.parallelism.trainer.SymmetricTrainer;
import org.deeplearning4j.parallelism.trainer.Trainer;

/**
 * Creates {@link DefaultTrainer}
 * instances for use with {@link ParallelWrapper}
 * @author raver119@gmail.com
 */
@Slf4j
public class SymmetricTrainerContext implements TrainerContext {
    /**
     * Initialize the context
     *
     * @param model
     * @param args the arguments to initialize with (maybe null)
     */
    @Override
    public void init(Model model, Object... args) {

    }

    /**
     * Create a {@link Trainer}
     * based on the given parameters
     *
     * @param threadId   the thread id to use for this worker
     * @param model      the model to start the trainer with
     * @param rootDevice the root device id
     * @param useMDS     whether to use the {@link MagicQueue}
     *                   or not
     * @param wrapper    the wrapper instance to use with this trainer (this refernece is needed
     *                   for coordination with the {@link ParallelWrapper} 's {@link TrainingListener}
     * @return the created training instance
     */
    @Override
    public Trainer create(String uuid, int threadId, Model model, int rootDevice, boolean useMDS, ParallelWrapper wrapper,
                    WorkspaceMode mode, int averagingFrequency) {

        SymmetricTrainer trainer = new SymmetricTrainer(model, uuid, threadId, mode, wrapper, useMDS);

        trainer.setName("SymmetricTrainer thread " + threadId);
        trainer.setDaemon(true);

        return trainer;
    }

    @Override
    public void finalizeRound(Model originalModel, Model... models) {
        // no-op
    }

    @Override
    public void finalizeTraining(Model originalModel, Model... models) {
        // we CAN avarage here, but for now we'll just push first model params to original model
        originalModel.setParams(models[0].params());
    }
}
