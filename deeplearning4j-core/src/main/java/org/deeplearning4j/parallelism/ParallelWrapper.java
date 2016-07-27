package org.deeplearning4j.parallelism;

import lombok.NonNull;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This is simple data-parallel wrapper suitable for multi-cpu/multi-gpu environments.
 *
 * @author raver119@gmail.com
 */
public class ParallelWrapper {
    private static Logger logger = LoggerFactory.getLogger(ParallelWrapper.class);
    private Model model;
    private int workers = 2;
    private int prefetchSize = 2;
    private int averagingFrequency = 1;
    private Trainer zoo[];
    private AtomicLong iterationsCounter = new AtomicLong(0);
    private boolean reportScore = false;

    protected ParallelWrapper(Model model, int workers, int prefetchSize) {
        this.model = model;
        this.workers = workers;
        this.prefetchSize = prefetchSize;

        zoo = new Trainer[workers];
        for (int cnt = 0; cnt < workers; cnt++) {
            zoo[cnt] = new Trainer(cnt, model);
            zoo[cnt].start();
        }
    }

    /**
     * This method takes DataSetIterator, and starts training over it by scheduling DataSets to different executors
     *
     * @param source
     */
    public synchronized void fit(@NonNull DataSetIterator source) {
        DataSetIterator iterator;
        if (prefetchSize > 0 && (!(source instanceof AsyncDataSetIterator) && !(source instanceof ListDataSetIterator))) {
            iterator = new AsyncDataSetIterator(source, prefetchSize);
        } else iterator = source;

        AtomicInteger locker = new AtomicInteger(0);


        iterator.reset();
        while (iterator.hasNext()) {
            DataSet dataSet = iterator.next();

            /*
             now dataSet should be dispatched to next free workers, until all workers are busy. And then we should block till all finished.
            */
            int pos = locker.getAndIncrement();
            zoo[pos].feedDataSet(dataSet);

            /*
                if all workers are dispatched now, join till all are finished
            */
            if (pos + 1 == workers || !iterator.hasNext()) {
                iterationsCounter.incrementAndGet();

                for (int cnt = 0; cnt < workers && cnt < locker.get(); cnt ++) {
                    try {
                        zoo[cnt].waitTillRunning();
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }


                /*
                    average model, and propagate it to whole
                */
                if (iterationsCounter.get() % averagingFrequency == 0 || !iterator.hasNext()) {
                    double score = 0.0;
                    List<INDArray> params = new ArrayList<>();
                    for (int cnt = 0; cnt < workers && cnt < locker.get(); cnt++) {
                        params.add(zoo[cnt].getModel().params());
                        score += zoo[cnt].getModel().score();
                    }
                    Nd4j.averageAndPropagate(model.params(), params);

                    score /= Math.min(workers, locker.get());

                    // TODO: improve this
                    if (reportScore)
                        logger.info("Averaged score: " + score);

                    if (model instanceof MultiLayerNetwork) {
                        Updater updater = ((MultiLayerNetwork)zoo[0].getModel()).getUpdater();


                        INDArray updaterState = Nd4j.zeros(updater.getStateViewArray().shape());

                        for( int cnt = 0; cnt< workers && cnt < locker.get(); cnt++ ){
                            Updater u = ((MultiLayerNetwork)zoo[cnt].getModel()).getUpdater();
                            INDArray updaterView = u.getStateViewArray();
                            updaterState.addi(updaterView);
                        }
                        updaterState.divi(Math.min(workers, locker.get()));
                        ((MultiLayerNetwork) model).getUpdater().setStateViewArray((Layer)model, updaterState, false);

                        ((MultiLayerNetwork) model).setScore(score);
                    } else if (model instanceof ComputationGraph) {
                        ComputationGraphUpdater updater = ((ComputationGraph)zoo[0].getModel()).getUpdater();

                        INDArray updaterState = Nd4j.zeros(updater.getStateViewArray().shape());

                        for( int cnt = 0; cnt< workers && cnt < locker.get(); cnt++ ){
                            ComputationGraphUpdater u = ((ComputationGraph)zoo[cnt].getModel()).getUpdater();
                            INDArray updaterView = u.getStateViewArray();
                            updaterState.addi(updaterView);
                        }
                        updaterState.divi(Math.min(workers, locker.get()));
                        ((ComputationGraph) model).getUpdater().setStateViewArray(updaterState);

                        ((ComputationGraph) model).setScore(score);
                    }

                    // FIXME: updateModel() call should be removed
                    for (int i = 0; i < workers; i++) {
                        zoo[i].updateModel(model);
                    }

                }
                locker.set(0);
            }
        }
    }

    public static class Builder {
        private Model model;
        private int workers = 2;
        private int prefetchSize = 2;
        private int averagingFrequency = 1;
        private boolean reportScore = false;

        /**
         * Build ParallelWrapper for MultiLayerNetwork
         *
         * @param mln
         */
        public Builder(@NonNull MultiLayerNetwork mln) {
            model = mln;
        }

        /**
         * Build ParallelWrapper for ComputationGraph
         *
         * @param graph
         */
        public Builder(@NonNull ComputationGraph graph) {
            model = graph;
        }

        /**
         * This method allows to configure number of workers that'll be used for parallel training
         *
         * @param num
         * @return
         */
        public Builder workers(int num) {
            if (num < 1)
                throw new RuntimeException("Number of workers can't be lower then 1!");

            this.workers = num;
            return this;
        }

        /**
         * Model averaging frequency.
         *
         * @param freq number of iterations between averagin
         * @return
         */
        public Builder averagingFrequency(int freq) {
            this.averagingFrequency = freq;
            return this;
        }


        /**
         * Size of prefetch buffer that will be used for background data prefetching.
         * Usually it's better to keep this value equal to the number of workers.
         *
         * Default value: 2
         *
         * @param size 0 to disable prefetching, any positive number
         * @return
         */
        public Builder prefetchBuffer(int size) {
            if (size < 0)
                size = 0;

            this.prefetchSize = size;

            return this;
        }

        public Builder reportScoreAfterAveraging(boolean reallyReport) {
            this.reportScore = reallyReport;
            return this;
        }

        /**
         * This method returns ParallelWrapper instance
         *
         * @return
         */
        public ParallelWrapper build() {
            ParallelWrapper wrapper = new ParallelWrapper(model, workers, prefetchSize);
            wrapper.averagingFrequency = this.averagingFrequency;
            wrapper.reportScore = this.reportScore;

            return wrapper;
        }
    }

    private static class Trainer extends Thread implements Runnable {
        private Model originalModel;
        private Model replicatedModel;
        private LinkedBlockingQueue<DataSet> queue = new LinkedBlockingQueue<>();
        private AtomicInteger running = new AtomicInteger(0);
        private int threadId;

        public Trainer(int threadId, Model model) {
            this.threadId = threadId;
            this.setDaemon(true);

            this.originalModel = model;
            if (model instanceof MultiLayerNetwork) {
                this.replicatedModel = ((MultiLayerNetwork) model).clone();

                if (threadId != 0)
                    ((MultiLayerNetwork)this.replicatedModel).setListeners(new ArrayList<IterationListener>());
            } else if (model instanceof ComputationGraph) {
                this.replicatedModel = ((ComputationGraph) model).clone();

                if (threadId != 0)
                    ((ComputationGraph)this.replicatedModel).setListeners(new ArrayList<IterationListener>());
            }
        }

        public void feedDataSet(@NonNull DataSet dataSet) {
            running.incrementAndGet();
            queue.add(dataSet);
        }

        public Model getModel() {
            return replicatedModel;
        }

        public void updateModel(@NonNull Model model) {


            if (model instanceof MultiLayerNetwork) {
                replicatedModel = ((MultiLayerNetwork) model).clone();
//                replicatedModel.setParams(model.params().dup());
//                ((MultiLayerNetwork) replicatedModel).setUpdater(((MultiLayerNetwork)model).getUpdater().clone());
            } else if (model instanceof  ComputationGraph) {
                replicatedModel = ((ComputationGraph) model).clone();
            }
        }

        public boolean isRunning(){
            return running.get() == 0;
        }

        @Override
        public void run() {
            try {
                while (true) {
                    DataSet dataSet = queue.poll(1, TimeUnit.SECONDS);
                    if (dataSet != null) {
                        if (replicatedModel instanceof MultiLayerNetwork) {
                            ((MultiLayerNetwork) replicatedModel).fit(dataSet);
                        } else if (replicatedModel instanceof ComputationGraph) {
                            ((ComputationGraph) replicatedModel).fit(dataSet);
                        }
                        running.decrementAndGet();
                    }
                }
            } catch (Exception e) {
                //
            }
        }

        public void waitTillRunning() {
            while (running.get() != 0) {
                try {
                    Thread.sleep(10);
                } catch (Exception e) {
                    ;
                }
            }
        }
    }
}
