package org.nd4j.autodiff.listeners;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;

public interface Listener {

    void epochStart(SameDiff sd, At at);

    void epochEnd(SameDiff sd, At at);

    void iterationStart(SameDiff sd, At at, MultiDataSet data);

    void iterationDone(SameDiff sd, At at, MultiDataSet dataSet, Loss loss);

    void opExecution(SameDiff sd, At at, SameDiffOp op, DifferentialFunction df, INDArray[] outputs);

}
