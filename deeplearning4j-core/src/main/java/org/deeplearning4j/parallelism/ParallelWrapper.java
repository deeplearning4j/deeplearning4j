package org.deeplearning4j.parallelism;

import lombok.NonNull;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

/**
 * @author raver119@gmail.com
 */
public class ParallelWrapper {
    private Model model;

    protected ParallelWrapper() {
        //
    }

    /**
     * This method takes DataSetIterator, and starts training over it
     *
     * @param iterator
     */
    public void fit(@NonNull DataSetIterator iterator) {

    }

    public static class Builder {
        private Model model;

        public Builder(@NonNull MultiLayerNetwork mln) {
            model = mln;
        }

        public Builder(@NonNull ComputationGraph graph) {
            model = graph;
        }

        public ParallelWrapper build() {
            ParallelWrapper wrapper = new ParallelWrapper();

            return wrapper;
        }
    }
}
