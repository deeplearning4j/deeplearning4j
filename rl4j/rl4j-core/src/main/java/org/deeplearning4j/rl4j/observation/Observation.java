package org.deeplearning4j.rl4j.observation;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;

public class Observation {
    // TODO: Presently only a dummy container. Will contain observation channels when done.

    private final DataSet data;

    public Observation(INDArray[] data) {
        this(new org.nd4j.linalg.dataset.DataSet(Nd4j.concat(0, data), null));
    }

    // FIXME: Remove -- only used in unit tests
    public Observation(INDArray data) {
        this.data = new org.nd4j.linalg.dataset.DataSet(data, null);
    }

    private Observation(DataSet data) {
        this.data = data;
    }

    public Observation dup() {
        return new Observation(new org.nd4j.linalg.dataset.DataSet(data.getFeatures().dup(), null));
    }

    public INDArray getData() {
        return data.getFeatures();
    }
}
