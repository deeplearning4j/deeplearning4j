package org.deeplearning4j.nn.conf.dropout;

import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface DropoutHelper {

    boolean checkSupported();

    void applyDropout(INDArray inputActivations, INDArray resultArray, double dropoutInputRetainProb);

    void backprop(INDArray gradAtOutput, INDArray gradAtInput);


}

