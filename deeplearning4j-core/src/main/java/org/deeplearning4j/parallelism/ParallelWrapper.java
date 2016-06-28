package org.deeplearning4j.parallelism;

import lombok.NonNull;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * @author raver119@gmail.com
 */
public class ParallelWrapper {
    private Model model;
    private int workers = 2;
    private int prefetchSize = 2;
    private Model zoo[];

    protected ParallelWrapper(Model model, int workers, int prefetchSize) {
        this.model = model;
        this.workers = workers;
        this.prefetchSize = prefetchSize;

        zoo = new Model[workers];
    }

    protected void fit(@NonNull DataSet dataSet) {

    }

    /**
     * This method takes DataSetIterator, and starts training over it by scheduling DataSets to different executors
     *
     * @param source
     */
    public void fit(@NonNull DataSetIterator source) {
        DataSetIterator iterator;
        if (prefetchSize > 0 && (!(source instanceof AsyncDataSetIterator) && !(source instanceof ListDataSetIterator))) {
            iterator = new AsyncDataSetIterator(source, prefetchSize);
        } else iterator = source;


        iterator.reset();
        while (iterator.hasNext()) {
            DataSet dataSet = iterator.next();

            /*
             now dataSet should be dispatched to next free workers, until all workers are busy. And then we should block till all finished.
            */


            /*
                if all workers are dispatched now, join till all are finished
            */
        }
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
}
