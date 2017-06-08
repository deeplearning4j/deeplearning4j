package org.deeplearning4j.spark.parameterserver.pw;

import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.optimize.solvers.accumulation.CudaGradientsAccumulator;
import org.deeplearning4j.optimize.solvers.accumulation.MessageHandler;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.spark.parameterserver.iterators.VirtualDataSetIterator;
import org.deeplearning4j.spark.parameterserver.iterators.VirtualMultiDataSetIterator;
import org.deeplearning4j.spark.parameterserver.networking.WiredEncodingHandler;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingWorker;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;

import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This class maintains ParallelWrapper instance in Spark environment, and provides primitives for inter-executor
 * communication during training over partitions.
 *
 * @author raver119@gmail.com
 */
public class SharedTrainingWrapper {
    public static SharedTrainingWrapper INSTANCE = new SharedTrainingWrapper();
    protected ParallelWrapper wrapper;
    protected VirtualDataSetIterator iteratorDS;
    protected VirtualMultiDataSetIterator iteratorMDS;

    protected List<Iterator<DataSet>> iteratorsDS;
    protected List<Iterator<MultiDataSet>> iteratorsMDS;


    protected AtomicBoolean isFirst = new AtomicBoolean(false);

    protected SharedTrainingWrapper() {

        // instantiate some stuff here
        iteratorsDS = new CopyOnWriteArrayList<>();
        iteratorsMDS = new CopyOnWriteArrayList<>();
    }

    public static SharedTrainingWrapper getInstance() {
        return INSTANCE;
    }

    /**
     * This method registers given Iterable<DataSet> in VirtualDataSetIterator
     *
     * @param iterator
     */
    public void attachDS(Iterator<DataSet> iterator) {
        iteratorsDS.add(iterator);
    }

    /**
     * This method registers given Iterable<MultiDataSet> in VirtualMultiDataSetIterator
     *
     * @param iterator
     */
    public void attachMDS(Iterator<MultiDataSet> iterator) {
        iteratorsMDS.add(iterator);
    }

    public void run(SharedTrainingWorker worker) {
        /*
            first call instantiates pw, messenger etc, and gets in charge here.
         */
        if (isFirst.compareAndSet(false, true)) {
            // getting model from worker, and instantiating PW

            Model model = worker.getInitialModel();
            if (model == null)
                model = worker.getInitialModelGraph();

            if (model == null)
                throw new DL4JInvalidConfigException("No model was defined for training");

            // TODO: make threshold configurable here
            MessageHandler handler = new WiredEncodingHandler(1e-3);

            // this accumulator will provide sharing gradients over network, via WiredEncodedHandler
            CudaGradientsAccumulator accumulator = new CudaGradientsAccumulator.Builder(2)
                    .messageHandler(handler)
                    .encodingThreshold(1e-3)
                    .build();

            wrapper = new ParallelWrapper.Builder<>(model)
                    // TODO: we should define proper num workers here, better suiting current environment
                    .workers(4)
                    // TODO: add configuration for workspaceMode here, or derive it from model?
                    .workspaceMode(WorkspaceMode.SEPARATE)
                    .trainingMode(ParallelWrapper.TrainingMode.CUSTOM)
                    .gradientsAccumulator(accumulator)

                    // TODO: decide, what do we want wrt prefetch
                    .prefetchBuffer(2)
                    .build();

            // TODO: optionally we might be waiting until we have >1 splits delivered

            // now we're just calling for fit
            if (iteratorDS != null)
                wrapper.fit(iteratorDS);
            else if (iteratorMDS != null)
                wrapper.fit(iteratorMDS);
            else
                throw new DL4JInvalidConfigException("No iterators were defined for training");

        } else {
            // no-op i guess, or blocking call right here
        }
    }

    public void passDataSet(DataSet dataSet) {
        // we're going to save this dataset into VirtualDataSetIterator
    }

    public void passDataSet(MultiDataSet dataSet) {
        // we're going to save this dataset into VirtualMultiDataSetIterator
    }


    public void blockUntilFinished() {
        // do something and block
    }
}
