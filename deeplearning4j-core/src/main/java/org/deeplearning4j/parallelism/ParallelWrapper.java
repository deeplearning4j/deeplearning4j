package org.deeplearning4j.parallelism;

import lombok.NonNull;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * PLEASE NOTE: UNDER CONSTRUCTION, DO NOT USE THIS CLASS
 *
 * @author raver119@gmail.com
 */
@Deprecated
public class ParallelWrapper {
    private static Logger logger = LoggerFactory.getLogger(ParallelWrapper.class);
    private Model model;
    private int workers = 2;
    private int prefetchSize = 2;
    private Trainer zoo[];

    protected ParallelWrapper(Model model, int workers, int prefetchSize) {
        this.model = model;
        this.workers = workers;
        this.prefetchSize = prefetchSize;

        zoo = new Trainer[workers];
        for (int cnt = 0; cnt < workers; cnt++) {
            zoo[cnt] = new Trainer(model);
        }
    }

    protected void fit(@NonNull DataSet dataSet) {

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
            zoo[pos].updateModel(model);
            zoo[pos].start();

            /*
                if all workers are dispatched now, join till all are finished
            */
            if (pos + 1 == workers || !iterator.hasNext()) {
                for (int cnt = 0; cnt < workers && cnt < locker.get(); cnt ++) {
                    try {
                        zoo[cnt].join();
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }


                /*
                    average model, and propagate it to whole
                */

                double score = 0.0;
                INDArray result = Nd4j.create(model.params().shape());
                for (int cnt = 0; cnt < workers && cnt < locker.get(); cnt++) {
                    INDArray params = zoo[cnt].getModel().params();
                    result.addi(params);
                    score += zoo[cnt].getModel().score();
                }
                result.divi(Math.min(workers, locker.get()));
                model.setParams(result);
                score /= Math.min(workers, locker.get());

                logger.info("Score: " + score);

                if (model instanceof MultiLayerNetwork) {

                    ((MultiLayerNetwork) model).setScore(score);
                } else if (model instanceof ComputationGraph) {

                    ((ComputationGraph) model).setScore(score);
                }


                for (int cnt = 0; cnt < workers; cnt++) {
                    zoo[cnt] = new Trainer(model);
                }
                locker.set(0);
            }
        }
    }

    protected void averageModels(Model... models) {

    }

    public static class Builder {
        private Model model;
        private int workers = 2;
        private int prefetchSize = 2;

        public Builder(@NonNull MultiLayerNetwork mln) {
            model = mln;
        }

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

        public ParallelWrapper build() {
            ParallelWrapper wrapper = new ParallelWrapper(model, workers, prefetchSize);

            return wrapper;
        }
    }

    private static class Trainer extends Thread implements Runnable {
        private Model originalModel;
        private Model replicatedModel;
        private DataSet dataSet;

        public Trainer(Model model) {
            this.originalModel = model;
        }

        public void feedDataSet(@NonNull DataSet dataSet) {
            this.dataSet = dataSet;
        }

        public Model getModel() {
            return replicatedModel;
        }

        public void updateModel(@NonNull Model model) {
            this.originalModel = model;
            if (model instanceof MultiLayerNetwork) {
                this.replicatedModel = ((MultiLayerNetwork) model).clone();
            } else if (model instanceof ComputationGraph) {
                this.replicatedModel = ((ComputationGraph) model).clone();
            }
        }

        @Override
        public void run() {
            if (dataSet == null)
                throw new IllegalStateException("DataSet is NULL");

            if (originalModel instanceof MultiLayerNetwork) {
                ((MultiLayerNetwork) replicatedModel).fit(dataSet);
            } else if (originalModel instanceof ComputationGraph) {
                ((ComputationGraph) replicatedModel).fit(dataSet);
            }
        }
    }
}
