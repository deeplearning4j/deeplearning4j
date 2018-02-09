package org.deeplearning4j.largevis;

import org.nd4j.linalg.api.ndarray.INDArray;

public class LargeVisKnnRunner implements Runnable {
    private LargeVis largeVis;
    private int id;
    private int numWorkers;
    private int numNeighbors;
    public LargeVisKnnRunner(LargeVis largeVis,
                             int id,
                             int numWorkers,int numNeighbors) {
        this.largeVis = largeVis;
        this.id = id;
        this.numWorkers = numWorkers;
        this.numNeighbors = numNeighbors;

    }

    @Override
    public void run() {
        int low = id * largeVis.getVec().rows() / numWorkers;
        int hi = (id + 1) * largeVis.getVec().rows() / numWorkers;
        for(int i = low; i < hi; i++) {
           INDArray query =  largeVis.getRpTree().queryAll(largeVis.getVec().slice(i),(numNeighbors + 1 ));
        }
    }
}
