package org.nd4j.autodiff.samediff;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.learning.config.IUpdater;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * TrainingConfig is a simple configuration class for defining settings for training a {@link SameDiff} instance.<br>
 * It defines the following settings:<br>
 * <ul>
 *     <li>The {@link IUpdater} to use (i.e., {@link org.nd4j.linalg.learning.config.Adam}, {@link org.nd4j.linalg.learning.config.Nesterovs} etc.
 *     The IUpdater instance is also how the learning rate (or learning rate schedule) is set.</li>
 *     <li>The L1 and L2 regularization coefficients (set to 0.0 by default)</li>
 *     <li>The DataSet feature and label mapping - which defines how the feature/label arrays from the DataSet/MultiDataSet
 *     should be associated with SameDiff variables (usually placeholders)</li>
 *     <li>Optional: The names of the trainable parameters. The trainable parameters are inferred automatically if not set here, though
 *     can be overridden if some parameters should not be modified during training (or if the automatic inference of the trainable
 *     parameters is not suitable/correct)</li>
 * </ul>
 * The TrainingConfig instance also stores the iteration count and the epoch count - these values are updated during training
 * and are used for example in learning rate schedules.
 *
 * @author Alex Black
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class TrainingConfig {

    private IUpdater updater;
    private double l1;
    private double l2;
    private boolean minimize = true;
    private List<String> dataSetFeatureMapping;
    private List<String> dataSetLabelMapping;
    private List<String> dataSetFeatureMaskMapping;
    private List<String> dataSetLabelMaskMapping;
    private List<String> trainableParams;   //Will be inferred automatically if null
    private int iterationCount;
    private int epochCount;

    /**
     * Create a training configuration suitable for training a single input, single output network.<br>
     * See also the {@link Builder} for creating a TrainingConfig
     *
     * @param updater               The updater configuration to use
     * @param l1                    The L1 regularization coefficient for the trainable parameters (set 0.0 to disable L1 regularization)
     * @param l2                    The L2 regularization coefficient for the trainable parameters (set 0.0 to disable L2 regularization)
     * @param dataSetFeatureMapping The name of the placeholder/variable that should be set using the feature INDArray from the DataSet
     *                              (or the first/only feature from a MultiDataSet). For example, if the network input placeholder was
     *                              called "input" then this should be set to "input"
     * @param dataSetLabelMapping   The name of the placeholder/variable that should be set using the label INDArray from the DataSet
     *                              (or the first/only feature from a MultiDataSet). For example, if the network input placeholder was
     *                              called "input" then this should be set to "input"
     */
    public TrainingConfig(IUpdater updater, double l1, double l2, String dataSetFeatureMapping, String dataSetLabelMapping) {
        this(updater, l1, l2, true, Collections.singletonList(dataSetFeatureMapping), Collections.singletonList(dataSetLabelMapping),
                Collections.<String>emptyList(), Collections.<String>emptyList(), null);
    }

    /**
     * Create a training configuration suitable for training both single input/output and multi input/output networks.<br>
     * See also the {@link Builder} for creating a TrainingConfig
     *
     * @param updater                   The updater configuration to use
     * @param l1                        The L1 regularization coefficient for the trainable parameters (set 0.0 to disable L1 regularization)
     * @param l2                        The L2 regularization coefficient for the trainable parameters (set 0.0 to disable L2 regularization)
     * @param minimize                  Set to true if the loss function should be minimized (usually true). False to maximize
     * @param dataSetFeatureMapping     The name of the placeholders/variables that should be set using the feature INDArray(s) from the
     *                                  DataSet or MultiDataSet. For example, if the network had 2 inputs called "input1" and "input2"
     *                                  and the MultiDataSet features should be mapped with {@code MultiDataSet.getFeatures(0)->"input1"}
     *                                  and {@code MultiDataSet.getFeatures(1)->"input2"}, then this should be set to {@code List<>("input1", "input2")}.
     * @param dataSetLabelMapping       As per dataSetFeatureMapping, but for the DataSet/MultiDataSet labels
     * @param dataSetFeatureMaskMapping May be null. If non-null, the variables that the MultiDataSet feature mask arrays should be associated with.
     * @param dataSetLabelMaskMapping   May be null. If non-null, the variables that the MultiDataSet label mask arrays should be associated with.
     * @param trainableParams           May be null. If null: the set of trainable parameters will automatically be inferred from the SameDiff structure.
     *                                  If non-null, this defines the set of parameters that should be modified during training
     */
    public TrainingConfig(IUpdater updater, double l1, double l2, boolean minimize, List<String> dataSetFeatureMapping, List<String> dataSetLabelMapping,
                          List<String> dataSetFeatureMaskMapping, List<String> dataSetLabelMaskMapping, List<String> trainableParams) {
        this.updater = updater;
        this.l1 = l1;
        this.l2 = l2;
        this.minimize = minimize;
        this.dataSetFeatureMapping = dataSetFeatureMapping;
        this.dataSetLabelMapping = dataSetLabelMapping;
        this.dataSetFeatureMaskMapping = dataSetFeatureMaskMapping;
        this.dataSetLabelMaskMapping = dataSetLabelMaskMapping;
        this.trainableParams = trainableParams;
    }

    /**
     * Increment the iteration count by 1
     */
    public void incrementIterationCount(){
        iterationCount++;
    }

    /**
     * Increment the epoch count by 1
     */
    public void incrementEpochCount(){
        epochCount++;
    }

    public static Builder builder(){
        return new Builder();
    }

    /**
     * Get the index of the label array that the specified variable is associated with
     * @param s Name of the variable
     * @return The index of the label variable, or -1 if not found
     */
    public int labelIdx(String s){
        return dataSetLabelMapping.indexOf(s);
    }

    public static class Builder {

        private IUpdater updater;
        private double l1;
        private double l2;
        private boolean minimize = true;
        private List<String> dataSetFeatureMapping;
        private List<String> dataSetLabelMapping;
        private List<String> dataSetFeatureMaskMapping;
        private List<String> dataSetLabelMaskMapping;
        private List<String> trainableParams;   //Will be inferred automatically if null

        /**
         * Set the updater (such as {@link org.nd4j.linalg.learning.config.Adam}, {@link org.nd4j.linalg.learning.config.Nesterovs}
         * etc. This is also how the learning rate (or learning rate schedule) is set.
         * @param updater  Updater to set
         */
        public Builder updater(IUpdater updater){
            this.updater = updater;
            return this;
        }

        /**
         * Sets the L1 regularization coefficient for all trainable parameters. Must be >= 0.<br>
         * Default value: 0.0 (disabled)
         * @param l1 L1 regularization coefficient
         */
        public Builder l1(double l1){
            Preconditions.checkState(l1 >= 0, "L1 regularization coefficient must be >= 0. Got %s", l1);
            this.l1 = l1;
            return this;
        }

        /**
         * Sets the L2 regularization coefficient for all trainable parameters. Must be >= 0.<br>
         * Default value: 0.0 (disabled)
         * @param l2 L2 regularization coefficient
         */
        public Builder l2(double l2){
            Preconditions.checkState(l2 >= 0, "L2 regularization coefficient must be >= 0. Got %s", l2);
            this.l2 = l2;
            return this;
        }

        /**
         * Sets whether the loss function should be minimized (true) or maximized (false).<br>
         * The loss function is usually minimized in SGD.<br>
         * Default: true.
         * @param minimize True to minimize, false to maximize
         * @return
         */
        public Builder minimize(boolean minimize){
            this.minimize = minimize;
            return this;
        }

        /**
         * Set the name of the placeholders/variables that should be set using the feature INDArray(s) from the
         * DataSet or MultiDataSet. For example, if the network had 2 inputs called "input1" and "input2"
         * and the MultiDataSet features should be mapped with {@code MultiDataSet.getFeatures(0)->"input1"}
         * and {@code MultiDataSet.getFeatures(1)->"input2"}, then this should be set to {@code List<>("input1", "input2")}.
         *
         * @param dataSetFeatureMapping Name of the variables/placeholders that the feature arrays should be mapped to
         */
        public Builder dataSetFeatureMapping(String... dataSetFeatureMapping){
            return dataSetFeatureMapping(Arrays.asList(dataSetFeatureMapping));
        }

        /**
         * Set the name of the placeholders/variables that should be set using the feature INDArray(s) from the
         * DataSet or MultiDataSet. For example, if the network had 2 inputs called "input1" and "input2"
         * and the MultiDataSet features should be mapped with {@code MultiDataSet.getFeatures(0)->"input1"}
         * and {@code MultiDataSet.getFeatures(1)->"input2"}, then this should be set to {@code "input1", "input2"}.
         *
         * @param dataSetFeatureMapping Name of the variables/placeholders that the feature arrays should be mapped to
         */
        public Builder dataSetFeatureMapping(List<String> dataSetFeatureMapping){
            Preconditions.checkNotNull(dataSetFeatureMapping != null && dataSetFeatureMapping.size() > 0, "No feature mapping was provided");
            this.dataSetFeatureMapping = dataSetFeatureMapping;
            return this;
        }

        /**
         * Set the name of the placeholders/variables that should be set using the labels INDArray(s) from the
         * DataSet or MultiDataSet. For example, if the network had 2 labels called "label1" and "label2"
         * and the MultiDataSet labels should be mapped with {@code MultiDataSet.getLabel(0)->"label1"}
         * and {@code MultiDataSet.getLabels(1)->"label"}, then this should be set to {@code "label1", "label2"}.
         *
         * @param dataSetLabelMapping Name of the variables/placeholders that the label arrays should be mapped to
         */
        public Builder dataSetLabelMapping(String... dataSetLabelMapping){
            return dataSetLabelMapping(Arrays.asList(dataSetLabelMapping));
        }

        /**
         * Set the name of the placeholders/variables that should be set using the labels INDArray(s) from the
         * DataSet or MultiDataSet. For example, if the network had 2 labels called "label1" and "label2"
         * and the MultiDataSet labels should be mapped with {@code MultiDataSet.getLabel(0)->"label1"}
         * and {@code MultiDataSet.getLabels(1)->"label"}, then this should be set to {@code "label1", "label2"}.
         *
         * @param dataSetLabelMapping Name of the variables/placeholders that the label arrays should be mapped to
         */
        public Builder dataSetLabelMapping(List<String> dataSetLabelMapping){
            Preconditions.checkNotNull(dataSetLabelMapping != null && dataSetLabelMapping.size() > 0, "No label mapping was provided");
            this.dataSetLabelMapping = dataSetLabelMapping;
            return this;
        }

        /**
         * Set the name of the placeholders/variables that should be set using the feature mask INDArray(s) from the
         * DataSet or MultiDataSet. For example, if the network had 2 mask variables called "mask1" and "mask2"
         * and the MultiDataSet features masks should be mapped with {@code MultiDataSet.getFeatureMaskArray(0)->"mask1"}
         * and {@code MultiDataSet.getFeatureMaskArray(1)->"mask2"}, then this should be set to {@code "mask2", "mask2"}.
         *
         * @param dataSetFeatureMaskMapping Name of the variables/placeholders that the feature arrays should be mapped to
         */
        public Builder dataSetFeatureMaskMapping(List<String> dataSetFeatureMaskMapping){
            this.dataSetFeatureMaskMapping = dataSetFeatureMaskMapping;
            return this;
        }

        /**
         * Set the name of the placeholders/variables that should be set using the label mask INDArray(s) from the
         * DataSet or MultiDataSet. For example, if the network had 2 mask variables called "mask1" and "mask2"
         * and the MultiDataSet label masks should be mapped with {@code MultiDataSet.getLabelMaskArray(0)->"mask1"}
         * and {@code MultiDataSet.getLabelMaskArray(1)->"mask2"}, then this should be set to {@code "mask2", "mask2"}.
         *
         * @param dataSetLabelMaskMapping Name of the variables/placeholders that the feature arrays should be mapped to
         */
        public Builder dataSetLabelMaskMapping(List<String> dataSetLabelMaskMapping){
            this.dataSetLabelMaskMapping = dataSetLabelMaskMapping;
            return this;
        }

        /**
         * Define the set of trainable parameters for the network.<br>
         * The trainable parameters are not set by default, which means they will be inferred automatically.<br>
         * The set of trainable parameters (variables) can be set here - any excluded from being set here won't be
         * modified during training
         * @param trainableParams Set of parameters/variables to train
         */
        public Builder trainableParams(String... trainableParams){
            return trainableParams(Arrays.asList(trainableParams));
        }

        /**
         * Define the set of trainable parameters for the network.<br>
         * The trainable parameters are not set by default, which means they will be inferred automatically.<br>
         * The set of trainable parameters (variables) can be set here - any excluded from being set here won't be
         * modified during training
         * @param trainableParams Set of parameters/variables to train
         */
        public Builder trainableParams(List<String> trainableParams) {
            this.trainableParams = trainableParams;
            return this;
        }

        public TrainingConfig build(){
            Preconditions.checkState(updater != null, "Updater (optimizer) must not be null. Use updater(IUpdater) to set an updater");
            Preconditions.checkState(dataSetFeatureMapping != null, "No DataSet feature mapping has been provided. A " +
                    "mapping between DataSet array positions and variables/placeholders must be provided - use  dateSetFeatureMapping(...) to set this");
            Preconditions.checkState(dataSetLabelMapping != null, "No DataSet label mapping has been provided. A " +
                    "mapping between DataSet array positions and variables/placeholders must be provided - use  dateSetLabelMapping(...) to set this");

            return new TrainingConfig(updater, l1, l2, minimize, dataSetFeatureMapping, dataSetLabelMapping,
                    dataSetFeatureMaskMapping, dataSetLabelMaskMapping, trainableParams);
        }
    }

}
