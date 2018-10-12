package org.nd4j.autodiff.samediff;

import lombok.Data;
import org.nd4j.base.Preconditions;

import java.util.Arrays;
import java.util.List;

@Data
public class TrainingConfig {

    private double l1;
    private double l2;
    private boolean minimize = true;
    private List<String> dataSetFeatureMapping;
    private List<String> dataSetLabelMapping;
    private List<String> trainableParams;   //Will be inferred automatically if null
    private int iterationCount;
    private int epochCount;

    public TrainingConfig(double l1, double l2, boolean minimize, List<String> dataSetFeatureMapping, List<String> dataSetLabelMapping,
                          List<String> trainableParams) {
        this.l1 = l1;
        this.l2 = l2;
        this.minimize = minimize;
        this.dataSetFeatureMapping = dataSetFeatureMapping;
        this.dataSetLabelMapping = dataSetLabelMapping;
        this.trainableParams = trainableParams;
    }


    public void incrementIterationCount(){
        iterationCount++;
    }

    public void incrementEpochCount(){
        epochCount++;
    }

    public static class Builder {

        private double l1;
        private double l2;
        private boolean minimize = true;
        private List<String> dataSetFeatureMapping;
        private List<String> dataSetLabelMapping;
        private List<String> trainableParams;   //Will be inferred automatically if null

        public Builder l1(double l1){
            Preconditions.checkState(l1 >= 0, "L1 regularization coefficient must be >= 0. Got %s", l1);
            this.l1 = l1;
            return this;
        }

        public Builder l2(double l2){
            Preconditions.checkState(l2 >= 0, "L2 regularization coefficient must be >= 0. Got %s", l2);
            this.l2 = l2;
            return this;
        }

        public Builder minimize(boolean minimize){
            this.minimize = minimize;
            return this;
        }

        public Builder dataSetFeatureMapping(String... featureMapping){
            return dataSetFeatureMapping(Arrays.asList(featureMapping));
        }

        public Builder dataSetFeatureMapping(List<String> dataSetFeatureMapping){
            Preconditions.checkNotNull(dataSetFeatureMapping != null && dataSetFeatureMapping.size() > 0, "No feature mapping was provided");
            this.dataSetFeatureMapping = dataSetFeatureMapping;
            return this;
        }

        public Builder dataSetLabelMapping(String... labelMapping){
            return dataSetLabelMapping(Arrays.asList(labelMapping));
        }

        public Builder dataSetLabelMapping(List<String> dataSetLabelMapping){
            Preconditions.checkNotNull(dataSetLabelMapping != null && dataSetLabelMapping.size() > 0, "No label mapping was provided");
            this.dataSetLabelMapping = dataSetLabelMapping;
            return this;
        }

        public Builder trainableParams(String... trainableParams){
            return trainableParams(Arrays.asList(trainableParams));
        }

        public Builder trainableParams(List<String> trainableParams){
            this.trainableParams = trainableParams;
            return this;
        }

        public TrainingConfig build(){
            Preconditions.checkState(dataSetFeatureMapping != null, "No DataSet feature mapping has been provided. A label mapping must be provided - use  ");
            Preconditions.checkState(dataSetFeatureMapping != null, "No DataSet feature mapping has been provided. Call ");

            return new TrainingConfig(l1, l2, minimize, dataSetFeatureMapping, dataSetLabelMapping, trainableParams);
        }
    }

}
