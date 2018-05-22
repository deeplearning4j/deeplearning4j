package org.deeplearning4j.parallelism.parameterserver;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.parallelism.trainer.DefaultTrainer;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.parameterserver.client.ParameterServerClient;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Using an {@link ParameterServerClient}
 * we maintain updates for training a neural net.
 * Training happens relative to the mode of the remote {@link org.nd4j.parameterserver.node.ParameterServerNode}
 *
 * @author Adam Gibson
 */
@Builder
@Slf4j
@AllArgsConstructor
@NoArgsConstructor
public class ParameterServerTrainer extends DefaultTrainer {
    private ParameterServerClient parameterServerClient;

    @Override
    public void feedMultiDataSet(@NonNull MultiDataSet dataSet, long time) {
        // FIXME: this is wrong, and should be fixed

        if (getModel() instanceof ComputationGraph) {
            ComputationGraph computationGraph = (ComputationGraph) getModel();
            computationGraph.fit(dataSet);
        } else {
            throw new IllegalArgumentException("MultiLayerNetworks can't fit multi datasets");
        }

        log.info("Sending parameters");
        //send the updated params
        parameterServerClient.pushNDArray(getModel().params());
    }

    @Override
    public void feedDataSet(@NonNull DataSet dataSet, long time) {
        // FIXME: this is wrong, and should be fixed. Training should happen within run() loop

        if (getModel() instanceof ComputationGraph) {
            ComputationGraph computationGraph = (ComputationGraph) getModel();
            computationGraph.fit(dataSet);
        } else {
            MultiLayerNetwork multiLayerNetwork = (MultiLayerNetwork) getModel();
            log.info("Calling fit on multi layer network");
            multiLayerNetwork.fit(dataSet);

        }

        log.info("About to send params in");
        //send the updated params
        parameterServerClient.pushNDArray(getModel().params());
        log.info("Sent params");
    }

    @Override
    public Model getModel() {
        return super.getModel();
    }

    @Override
    public void updateModel(@NonNull Model model) {
        super.updateModel(model);
    }

    public static class ParameterServerTrainerBuilder extends DefaultTrainerBuilder {
        @Override
        public ParameterServerTrainerBuilder originalModel(Model originalModel) {
            return (ParameterServerTrainerBuilder) super.originalModel(originalModel);
        }

        @Override
        public ParameterServerTrainerBuilder replicatedModel(Model replicatedModel) {
            return (ParameterServerTrainerBuilder) super.replicatedModel(replicatedModel);
        }

        @Override
        public ParameterServerTrainerBuilder queue(LinkedBlockingQueue<DataSet> queue) {
            return (ParameterServerTrainerBuilder) super.queue(queue);
        }

        @Override
        public ParameterServerTrainerBuilder queueMDS(LinkedBlockingQueue<MultiDataSet> queueMDS) {
            return (ParameterServerTrainerBuilder) super.queueMDS(queueMDS);
        }

        @Override
        public ParameterServerTrainerBuilder running(AtomicInteger running) {
            return (ParameterServerTrainerBuilder) super.running(running);
        }

        @Override
        public ParameterServerTrainerBuilder threadId(int threadId) {
            return (ParameterServerTrainerBuilder) super.threadId(threadId);
        }

        @Override
        public ParameterServerTrainerBuilder shouldUpdate(AtomicBoolean shouldUpdate) {
            return (ParameterServerTrainerBuilder) super.shouldUpdate(shouldUpdate);
        }

        @Override
        public ParameterServerTrainerBuilder shouldStop(AtomicBoolean shouldStop) {
            return (ParameterServerTrainerBuilder) super.shouldStop(shouldStop);
        }

        @Override
        public ParameterServerTrainerBuilder thrownException(Exception thrownException) {
            return (ParameterServerTrainerBuilder) super.thrownException(thrownException);
        }

        @Override
        public ParameterServerTrainerBuilder useMDS(boolean useMDS) {
            return (ParameterServerTrainerBuilder) super.useMDS(useMDS);
        }

        @Override
        public ParameterServerTrainerBuilder onRootModel(boolean onRootModel) {
            return (ParameterServerTrainerBuilder) super.onRootModel(onRootModel);
        }

        @Override
        public ParameterServerTrainerBuilder parallelWrapper(ParallelWrapper parallelWrapper) {
            return (ParameterServerTrainerBuilder) super.parallelWrapper(parallelWrapper);
        }

        @Override
        public ParameterServerTrainerBuilder averagingFrequency(int frequency) {
            return (ParameterServerTrainerBuilder) super.averagingFrequency(frequency);
        }
    }
}
