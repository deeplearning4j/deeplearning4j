package org.deeplearning4j.spark.impl.computationgraph.dataset;

import org.apache.spark.api.java.function.Function;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;

/**Convert a DataSet to a MultiDataSet
 */
public class DataSetToMultiDataSetFn implements Function<DataSet,MultiDataSet> {
    @Override
    public MultiDataSet call(DataSet d) throws Exception {
        if(d.hasMaskArrays()){
            INDArray[] features = new INDArray[]{d.getFeatures()};
            INDArray[] labels = new INDArray[]{d.getLabels()};
            INDArray[] fmask = (d.getFeaturesMaskArray() != null ? new INDArray[]{d.getFeaturesMaskArray()} : null);
            INDArray[] lmask = (d.getLabelsMaskArray() != null ? new INDArray[]{d.getLabelsMaskArray()} : null);
            return new org.nd4j.linalg.dataset.MultiDataSet(features,labels,fmask,lmask);
        } else {
            return new org.nd4j.linalg.dataset.MultiDataSet(d.getFeatures(),d.getLabels());
        }

    }
}
