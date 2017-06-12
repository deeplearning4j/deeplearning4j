package org.deeplearning4j.spark.parameterserver.pw;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.optimize.solvers.accumulation.CudaGradientsAccumulator;
import org.deeplearning4j.optimize.solvers.accumulation.MessageHandler;
import org.deeplearning4j.parallelism.ParallelWrapper;
import org.deeplearning4j.spark.parameterserver.conf.SharedTrainingConfiguration;
import org.deeplearning4j.spark.parameterserver.iterators.VirtualDataSetIterator;
import org.deeplearning4j.spark.parameterserver.iterators.VirtualIterator;
import org.deeplearning4j.spark.parameterserver.iterators.VirtualMultiDataSetIterator;
import org.deeplearning4j.spark.parameterserver.networking.SilentTrainingDriver;
import org.deeplearning4j.spark.parameterserver.networking.WiredEncodingHandler;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingWorker;
import org.deeplearning4j.spark.parameterserver.util.BlockingObserver;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.parameterserver.distributed.VoidParameterServer;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.nd4j.parameterserver.distributed.enums.TransportType;
import org.nd4j.parameterserver.distributed.transport.MulticastTransport;
import org.nd4j.parameterserver.distributed.transport.RoutedTransport;
import org.nd4j.parameterserver.distributed.transport.Transport;

import java.util.Iterator;
import java.util.List;
import java.util.Observer;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This class maintains ParallelWrapper instance in Spark environment, and provides primitives for inter-executor
 * communication during training over partitions.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class SharedTrainingWrapper {
    public static SharedTrainingWrapper INSTANCE = new SharedTrainingWrapper();
    protected ParallelWrapper wrapper;
    protected VirtualDataSetIterator iteratorDS;
    protected VirtualMultiDataSetIterator iteratorMDS;

    protected List<Iterator<DataSet>> iteratorsDS;
    protected List<Iterator<MultiDataSet>> iteratorsMDS;


    protected AtomicBoolean isFirst = new AtomicBoolean(false);

    protected ThreadLocal<BlockingObserver> observer = new ThreadLocal<>();

    protected SharedTrainingWrapper() {

        // instantiate some stuff here
        iteratorsDS = new CopyOnWriteArrayList<>();
        iteratorsMDS = new CopyOnWriteArrayList<>();

        // now we're creating DataSetIterators, to feed ParallelWrapper
        iteratorDS = new VirtualDataSetIterator(iteratorsDS);
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
        // we're creating our Observable wrapper
        VirtualIterator<DataSet> wrapped = new VirtualIterator<>(iterator);

        // and creating Observer which will be used to monitor progress within iterator
        BlockingObserver obs = new BlockingObserver();
        wrapped.addObserver(obs);

        // putting that "somewhere"
        iteratorsDS.add(wrapped);

        // storing observer into ThreadLocal, since we're going to use that later
        observer.set(obs);
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
            log.info("Starting ParallelWrapper at thread {}", Thread.currentThread().getId());

            SharedTrainingConfiguration trainingConfiguration = worker.getBroadcastConfiguration().getValue();
            VoidConfiguration voidConfiguration = worker.getBroadcastConfiguration().getValue().getVoidConfiguration();

            Model model = worker.getInitialModel();
            if (model == null)
                model = worker.getInitialModelGraph();

            if (model == null)
                throw new DL4JInvalidConfigException("No model was defined for training");

            MessageHandler handler = new WiredEncodingHandler(trainingConfiguration.getThreshold());

            // this accumulator will provide sharing gradients over network, via WiredEncodedHandler
            CudaGradientsAccumulator accumulator = new CudaGradientsAccumulator.Builder(2)
                    .messageHandler(handler)
                    .encodingThreshold(trainingConfiguration.getThreshold())
                    .build();


            // FIXME: implement support for Custom transport implementation
            Transport transport = voidConfiguration.getTransportType() == TransportType.ROUTED ? new RoutedTransport() : voidConfiguration.getTransportType() == TransportType.BROADCAST ? new MulticastTransport() : null;

            if (transport == null)
                throw new DL4JInvalidConfigException("No Transport implementation was defined for this training session!");

            // now we're attaching VoidParameterServer to GradientsAccumulator
            VoidParameterServer.getInstance().init(voidConfiguration, transport, new SilentTrainingDriver(accumulator));

            wrapper = new ParallelWrapper.Builder<>(model)
                    // TODO: we should define proper num workers here, better suiting current environment
                    .workers(4)
                    .workspaceMode(trainingConfiguration.getWorkspaceMode())
                    .trainingMode(ParallelWrapper.TrainingMode.CUSTOM)
                    .gradientsAccumulator(accumulator)
                    .prefetchBuffer(trainingConfiguration.getPrefetchSize())
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
            // blocking call right here, all non-master threads will be blocked here
            try {
                //observer.get().waitTillDone();
                observer.get().wait();
            } catch (InterruptedException e) {
                // FIXME: we don't really need to throw it again, it's here only for debugging purposes
                throw new RuntimeException(e);
            }
        }
    }

    public void passDataSet(DataSet dataSet) {
        // we're going to save this dataset into VirtualDataSetIterator
    }

    public void passDataSet(MultiDataSet dataSet) {
        // we're going to save this dataset into VirtualMultiDataSetIterator
    }


    public void blockUntilFinished() throws InterruptedException {
        if (observer.get() != null)
            observer.get().wait();
        else
            throw new IllegalStateException("This method can't be called before iterators initialization");
    }
}
