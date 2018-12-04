package org.deeplearning4j.spark.parameterserver.python;
import org.nd4j.linalg.dataset.DataSet;

import java.util.ArrayList;
import java.util.List;

public class DataSetDescriptor implements java.io.Serializable{
    private ArrayDescriptor features, labels;
    private ArrayDescriptor featuresMask;
    private ArrayDescriptor labelsMask;
    private transient boolean preProcessed = false;


    public DataSetDescriptor(DataSet ds){
        features = ArrayDescriptor(ds.getFeatures());
        labels = ArrayDescriptor(ds.getLabels());
        featuresMask = ArrayDescriptor(ds.getFeaturesMask());
        labelsMask = ArrayDescriptor(ds.getLabelsMask());
        preProcessed = ds.isPreProcessed();
    }

    public DataSet getDataSet(){
        INDArray features = this.features.getArray();
        INDArray labels = this.labels.getArray();
        INDArray featuresMask = this.labels.getArray();
        INDArray labelsMask = this.labelsMask.getArray();
        DataSet ds = new DataSet(features, labels, featuresMask, labelsMask);
        ds.markAsPreProcessed(preProcessed);
        return ds;
    }
}
