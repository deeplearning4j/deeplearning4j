package org.deeplearning4j.spark.parameterserver.python;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;


public class DataSetDescriptor implements java.io.Serializable{
    private ArrayDescriptor features, labels;
    private ArrayDescriptor featuresMask;
    private ArrayDescriptor labelsMask;
    private boolean preProcessed;


    public DataSetDescriptor(DataSet ds)throws Exception{
        features = new ArrayDescriptor(ds.getFeatures());
        labels = new ArrayDescriptor(ds.getLabels());
        featuresMask = new ArrayDescriptor(ds.getFeaturesMaskArray());
        labelsMask = new ArrayDescriptor(ds.getLabelsMaskArray());
        preProcessed = ds.isPreProcessed();
    }

    public DataSet getDataSet(){
        INDArray features = this.features.getArray();
        INDArray labels = this.labels.getArray();
        INDArray featuresMask = this.labels.getArray();
        INDArray labelsMask = this.labelsMask.getArray();
        DataSet ds = new DataSet(features, labels, featuresMask, labelsMask);
        if(preProcessed) {
            ds.markAsPreProcessed();
        }
        return ds;
    }
}
