package org.deeplearning4j.nn.graph.util;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

public class ComputationGraphUtil {

    /** Convert a DataSet to the equivalent MultiDataSet */
    public static MultiDataSet toMultiDataSet(DataSet dataSet){
        INDArray f = dataSet.getFeatureMatrix();
        INDArray l = dataSet.getLabels();
        INDArray fMask = dataSet.getFeaturesMaskArray();
        INDArray lMask = dataSet.getLabelsMaskArray();

        INDArray[] fNew = new INDArray[]{f};
        INDArray[] lNew = new INDArray[]{l};
        INDArray[] fMaskNew = (fMask != null ? new INDArray[]{fMask} : null);
        INDArray[] lMaskNew = (lMask != null ? new INDArray[]{lMask} : null);

        return new org.nd4j.linalg.dataset.MultiDataSet(fNew,lNew,fMaskNew,lMaskNew);
    }

}
