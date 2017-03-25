package org.deeplearning4j.parallelism.parameterserver;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.parallelism.MagicQueue;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.parallelism.factory.TrainerFactory;
import org.deeplearning4j.parallelism.trainer.Trainer;
import org.nd4j.parameterserver.client.ParameterServerClient;

/**
 * Created by agibsonccc on 3/24/17.
 */
public class ParameterServerTrainerFactory implements TrainerFactory {

    private ParameterServerClient parameterServerClient;


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
     *                   for coordination with the {@link ParallelWrapper} 's {@link IterationListener}
     * @return the created training instance
     */
    @Override
    public Trainer create(int threadId, Model model, int rootDevice, boolean useMDS, ParallelWrapper wrapper) {
        return ParameterServerTrainer.builder()
                .originalModel(model)
                .parameterServerClient(parameterServerClient)
                .replicatedModel(model)
                .threadId(threadId)
                .parallelWrapper(wrapper)
                .useMDS(useMDS).build();
    }
}
