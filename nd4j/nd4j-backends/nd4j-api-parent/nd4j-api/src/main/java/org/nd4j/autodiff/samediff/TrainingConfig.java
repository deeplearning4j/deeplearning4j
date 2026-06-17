/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.autodiff.samediff;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.config.LossScaleConfig;
import org.nd4j.common.base.Preconditions;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.regularization.L1Regularization;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.learning.regularization.WeightDecay;
import org.nd4j.serde.json.JsonMappers;

import java.io.IOException;
import java.util.*;

@Data
@NoArgsConstructor
@AllArgsConstructor
@Slf4j
public class TrainingConfig {

    private IUpdater updater;
    private List<Regularization> regularization = new ArrayList<>();    //Regularization for all trainable parameters
    private boolean minimize = true;
    private List<String> dataSetFeatureMapping;
    private List<String> dataSetLabelMapping;
    private List<String> dataSetFeatureMaskMapping;
    private List<String> dataSetLabelMaskMapping;
    private int iterationCount;
    private int epochCount;
    private DataType initialLossDataType;

    // Mixed precision training configuration
    private DataType computeDataType = DataType.FLOAT;      // Forward/backward compute dtype
    private DataType masterWeightDataType = DataType.FLOAT; // Master weight dtype (for FP32 master weights)
    private LossScaleConfig lossScaleConfig;                // Loss scaling configuration
    private int gradientAccumulationSteps = 1;              // Number of steps to accumulate gradients

    private Map<String, List<IEvaluation>> trainEvaluations = new HashMap<>();
    private Map<String, Integer> trainEvaluationLabels = new HashMap<>();

    private Map<String, List<IEvaluation>> validationEvaluations = new HashMap<>();
    private Map<String, Integer> validationEvaluationLabels = new HashMap<>();

    /**
     * Create a training configuration suitable for training a single input, single output network.<br>
     * See also the {@link Builder} for creating a TrainingConfig
     *
     * @param updater               The updater configuration to use
     * @param dataSetFeatureMapping The name of the placeholder/variable that should be set using the feature INDArray from the DataSet
     *                              (or the first/only feature from a MultiDataSet). For example, if the network input placeholder was
     *                              called "input" then this should be set to "input"
     * @param dataSetLabelMapping   The name of the placeholder/variable that should be set using the label INDArray from the DataSet
     *                              (or the first/only feature from a MultiDataSet). For example, if the network input placeholder was
     *                              called "input" then this should be set to "input"
     */
    public TrainingConfig(IUpdater updater, List<Regularization> regularization, String dataSetFeatureMapping, String dataSetLabelMapping) {
        this(updater, regularization, true, Collections.singletonList(dataSetFeatureMapping), Collections.singletonList(dataSetLabelMapping),
                Collections.<String>emptyList(), null, DataType.FLOAT);
    }

    /**
     * Create a training configuration suitable for training both single input/output and multi input/output networks.<br>
     * See also the {@link Builder} for creating a TrainingConfig
     *
     * @param updater                   The updater configuration to use
     * @param regularization            Regularization for all trainable parameters;\
     * @param minimize                  Set to true if the loss function should be minimized (usually true). False to maximize
     * @param dataSetFeatureMapping     The name of the placeholders/variables that should be set using the feature INDArray(s) from the
     *                                  DataSet or MultiDataSet. For example, if the network had 2 inputs called "input1" and "input2"
     *                                  and the MultiDataSet features should be mapped with {@code MultiDataSet.getFeatures(0)->"input1"}
     *                                  and {@code MultiDataSet.getFeatures(1)->"input2"}, then this should be set to {@code List<>("input1", "input2")}.
     * @param dataSetLabelMapping       As per dataSetFeatureMapping, but for the DataSet/MultiDataSet labels
     * @param dataSetFeatureMaskMapping May be null. If non-null, the variables that the MultiDataSet feature mask arrays should be associated with.
     * @param dataSetLabelMaskMapping   May be null. If non-null, the variables that the MultiDataSet label mask arrays should be associated with.
     */
    public TrainingConfig(IUpdater updater, List<Regularization> regularization, boolean minimize, List<String> dataSetFeatureMapping, List<String> dataSetLabelMapping,
                          List<String> dataSetFeatureMaskMapping, List<String> dataSetLabelMaskMapping, DataType initialLossDataType) {
        this.updater = updater;
        this.regularization = regularization;
        this.minimize = minimize;
        this.dataSetFeatureMapping = dataSetFeatureMapping;
        this.dataSetLabelMapping = dataSetLabelMapping;
        this.dataSetFeatureMaskMapping = dataSetFeatureMaskMapping;
        this.dataSetLabelMaskMapping = dataSetLabelMaskMapping;
        this.initialLossDataType = initialLossDataType;
    }

    protected TrainingConfig(IUpdater updater, List<Regularization> regularization, boolean minimize, List<String> dataSetFeatureMapping, List<String> dataSetLabelMapping,
            List<String> dataSetFeatureMaskMapping, List<String> dataSetLabelMaskMapping,
            Map<String, List<IEvaluation>> trainEvaluations, Map<String, Integer> trainEvaluationLabels,
            Map<String, List<IEvaluation>> validationEvaluations, Map<String, Integer> validationEvaluationLabels, DataType initialLossDataType,
            DataType computeDataType, DataType masterWeightDataType, LossScaleConfig lossScaleConfig, int gradientAccumulationSteps) {
        this(updater, regularization, minimize, dataSetFeatureMapping, dataSetLabelMapping, dataSetFeatureMaskMapping, dataSetLabelMaskMapping, initialLossDataType);
        this.trainEvaluations = trainEvaluations;
        this.trainEvaluationLabels = trainEvaluationLabels;
        this.validationEvaluations = validationEvaluations;
        this.validationEvaluationLabels = validationEvaluationLabels;
        this.computeDataType = computeDataType;
        this.masterWeightDataType = masterWeightDataType;
        this.lossScaleConfig = lossScaleConfig;
        this.gradientAccumulationSteps = gradientAccumulationSteps;
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

    /**
     * Check if mixed precision training is enabled.
     * Mixed precision is considered enabled if the compute data type differs from FLOAT.
     *
     * @return true if mixed precision training is enabled
     */
    public boolean isMixedPrecision() {
        return computeDataType != null && computeDataType != DataType.FLOAT && computeDataType != DataType.DOUBLE;
    }

    /**
     * Check if loss scaling is enabled.
     *
     * @return true if loss scaling configuration is set and enabled
     */
    public boolean isLossScalingEnabled() {
        return lossScaleConfig != null && lossScaleConfig.isEnabled();
    }

    /**
     * Check if gradient accumulation is enabled.
     *
     * @return true if gradient accumulation steps > 1
     */
    public boolean isGradientAccumulationEnabled() {
        return gradientAccumulationSteps > 1;
    }

    public static class Builder {

        private IUpdater updater;
        private List<Regularization> regularization = new ArrayList<>();
        private boolean minimize = true;
        private List<String> dataSetFeatureMapping;
        private List<String> dataSetLabelMapping;
        private List<String> dataSetFeatureMaskMapping;
        private List<String> dataSetLabelMaskMapping;
        private boolean skipValidation = false;
        private boolean markLabelsUnused = false;
        private DataType initialLossDataType = DataType.FLOAT;

        // Mixed precision training configuration
        private DataType computeDataType = DataType.FLOAT;
        private DataType masterWeightDataType = DataType.FLOAT;
        private LossScaleConfig lossScaleConfig;
        private int gradientAccumulationSteps = 1;

        private Map<String, List<IEvaluation>> trainEvaluations = new HashMap<>();
        private Map<String, Integer> trainEvaluationLabels = new HashMap<>();

        private Map<String, List<IEvaluation>> validationEvaluations = new HashMap<>();
        private Map<String, Integer> validationEvaluationLabels = new HashMap<>();


        /**
         * Set the initial loss data type, defaults to
         * {@link DataType#FLOAT} - when setting a data type for a loss function
         * we need a beginning data type to compute the gradients. In order to do so,
         * we need to set an initial number of zero that acts as the initial gradient.
         * This initial loss data type controls the data type of that number.
         * This is critical when wanting more fine grained control over the data types
         * used in the training process.
         * @param initialLossDataType the initial loss data type
         * @return this builder
         */
        public Builder initialLossDataType(DataType initialLossDataType) {
            this.initialLossDataType = initialLossDataType;
            return this;
        }

        /**
         * Set the data type used for forward and backward computations.
         * For mixed precision training, this is typically FLOAT16 or BFLOAT16.
         * Default: FLOAT
         *
         * @param computeDataType The data type for compute operations
         * @return this builder
         */
        public Builder computeDataType(DataType computeDataType) {
            this.computeDataType = computeDataType;
            return this;
        }

        /**
         * Set the data type used for master weights.
         * Master weights maintain full precision (usually FLOAT) during mixed precision training.
         * Default: FLOAT
         *
         * @param masterWeightDataType The data type for master weights
         * @return this builder
         */
        public Builder masterWeightDataType(DataType masterWeightDataType) {
            this.masterWeightDataType = masterWeightDataType;
            return this;
        }

        /**
         * Configure loss scaling for mixed precision training.
         * Loss scaling helps prevent gradient underflow when training with FP16.
         *
         * @param lossScaleConfig The loss scale configuration
         * @return this builder
         */
        public Builder lossScaling(LossScaleConfig lossScaleConfig) {
            this.lossScaleConfig = lossScaleConfig;
            return this;
        }

        /**
         * Set the number of gradient accumulation steps.
         * Gradients will be accumulated over this many mini-batches before
         * being applied. This allows effective larger batch sizes when memory
         * is limited.
         * Default: 1 (no accumulation)
         *
         * @param steps Number of accumulation steps (must be >= 1)
         * @return this builder
         */
        public Builder gradientAccumulationSteps(int steps) {
            if (steps < 1) {
                throw new IllegalArgumentException("gradientAccumulationSteps must be >= 1, got: " + steps);
            }
            this.gradientAccumulationSteps = steps;
            return this;
        }

        /**
         * Enable mixed precision training with FP16 compute and FP32 master weights.
         * This is a convenience method that sets:
         * - computeDataType to FLOAT16
         * - masterWeightDataType to FLOAT
         * - lossScaling to dynamic scaling with default parameters
         *
         * @return this builder
         */
        public Builder mixedPrecision() {
            this.computeDataType = DataType.FLOAT16;
            this.masterWeightDataType = DataType.FLOAT;
            this.lossScaleConfig = LossScaleConfig.dynamicScaling();
            return this;
        }

        /**
         * Enable mixed precision training with BFLOAT16 compute and FP32 master weights.
         * BFLOAT16 has the same dynamic range as FP32, so loss scaling is typically not needed.
         *
         * @return this builder
         */
        public Builder mixedPrecisionBfloat16() {
            this.computeDataType = DataType.BFLOAT16;
            this.masterWeightDataType = DataType.FLOAT;
            this.lossScaleConfig = null; // BF16 typically doesn't need loss scaling
            return this;
        }

        /**
         * Set the updater (such as {@link org.nd4j.linalg.learning.config.Adam}, {@link org.nd4j.linalg.learning.config.Nesterovs}
         * etc. This is also how the learning rate (or learning rate schedule) is set.
         * @param updater  Updater to set
         */
        public Builder updater(IUpdater updater) {
            this.updater = updater;
            return this;
        }


        /**
         * Sets the L1 regularization coefficient for all trainable parameters. Must be >= 0.<br>
         * See {@link L1Regularization} for more details
         * @param l1 L1 regularization coefficient
         */
        public Builder l1(double l1) {
            Preconditions.checkState(l1 >= 0, "L1 regularization coefficient must be >= 0. Got %s", l1);
            removeInstances(this.regularization, L1Regularization.class);
            this.regularization.add(new L1Regularization(l1));
            return this;
        }

        /**
         Sets the L2 regularization coefficient for all trainable parameters. Must be >= 0.<br>
         * <b>Note</b>: Generally, {@link WeightDecay} (set via {@link #weightDecay(double,boolean)} should be preferred to
         * L2 regularization. See {@link WeightDecay} javadoc for further details.<br>
         * Note: L2 regularization and weight decay usually should not be used together; if any weight decay (or L2) has
         * been added for the biases, these will be removed first.
         *
         * @see #weightDecay(double, boolean)
         */
        public Builder l2(double l2){
            Preconditions.checkState(l2 >= 0.0, "L2 regularization coefficient must be >= 0. Got %s", l2);
            //Check if existing L2 exists; if so, replace it. Also remove weight decay - it doesn't make sense to use both
            removeInstances(this.regularization, L2Regularization.class);
            if(l2 > 0.0) {
                removeInstancesWithWarning(this.regularization, WeightDecay.class, "WeightDecay regularization removed: incompatible with added L2 regularization");
                this.regularization.add(new L2Regularization(l2));
            }
            return this;
        }

        /**
         * Add weight decay regularization for all trainable parameters. See {@link WeightDecay} for more details.<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.<br>
         *
         * @param coefficient Weight decay regularization coefficient
         * @param applyLR     Whether the learning rate should be multiplied in when performing weight decay updates. See {@link WeightDecay} for more details.
         */
        public Builder weightDecay(double coefficient, boolean applyLR) {
            //Check if existing weight decay if it exists; if so, replace it. Also remove L2 - it doesn't make sense to use both
            removeInstances(this.regularization, WeightDecay.class);
            if(coefficient > 0.0) {
                removeInstancesWithWarning(this.regularization, L2Regularization.class, "L2 regularization removed: incompatible with added WeightDecay regularization");
                this.regularization.add(new WeightDecay(coefficient, applyLR));
            }
            return this;
        }

        /**
         * Add regularization to all trainable parameters in the network
         *
         * @param regularizations Regularization type(s) to add
         */
        public Builder addRegularization(Regularization... regularizations){
            Collections.addAll(this.regularization, regularizations);
            return this;
        }

        /**
         * Set the regularization for all trainable parameters in the network.
         * Note that if any existing regularization types have been added, they will be removed
         *
         * @param regularization Regularization type(s) to add
         */
        public Builder regularization(Regularization... regularization){
            if(regularization == null || regularization.length == 0)
                return this;
            List<Regularization> r = new ArrayList<>();
            Collections.addAll(r, regularization);
            return regularization(r);
        }

        /**
         * Set the regularization for all trainable parameters in the network.
         * Note that if any existing regularization types have been added, they will be removed
         *
         * @param regularization Regularization type(s) to add
         */
        public Builder regularization(List<Regularization> regularization){
            this.regularization = regularization;
            return this;
        }

        /**
         * Sets whether the loss function should be minimized (true) or maximized (false).<br>
         * The loss function is usually minimized in SGD.<br>
         * Default: true.
         * @param minimize True to minimize, false to maximize
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
         * Calling this method will mark the label as unused. This is basically a way to turn off label mapping validation in
         * TrainingConfig builder, for training models without labels.<br>
         * Put another way: usually you need to call {@link #dataSetLabelMapping(String...)} to set labels, this method
         * allows you to say that the DataSet/MultiDataSet labels aren't used in training.
         */
        public Builder markLabelsUnused(){
            this.markLabelsUnused = true;
            return this;
        }

        /**
         * See {@link #dataSetFeatureMaskMapping(List)}
         */
        public Builder dataSetFeatureMaskMapping(String... dataSetFeatureMaskMapping){
            return dataSetFeatureMaskMapping(Arrays.asList(dataSetFeatureMaskMapping));
        }

        /**
         * Set the name of the placeholders/variables that should be set using the feature mask INDArray(s) from the
         * DataSet or MultiDataSet. For example, if the network had 2 mask variables called "mask1" and "mask2"
         * and the MultiDataSet features masks should be mapped with {@code MultiDataSet.getFeatureMaskArray(0)->"mask1"}
         * and {@code MultiDataSet.getFeatureMaskArray(1)->"mask2"}, then this should be set to {@code "mask1", "mask2"}.
         *
         * @param dataSetFeatureMaskMapping Name of the variables/placeholders that the feature arrays should be mapped to
         */
        public Builder dataSetFeatureMaskMapping(List<String> dataSetFeatureMaskMapping){
            this.dataSetFeatureMaskMapping = dataSetFeatureMaskMapping;
            return this;
        }

        /**
         * See {@link #dataSetLabelMaskMapping(List)}
         */
        public Builder dataSetLabelMaskMapping(String... dataSetLabelMaskMapping){
            return dataSetLabelMaskMapping(Arrays.asList(dataSetLabelMaskMapping));
        }

        /**
         * Set the name of the placeholders/variables that should be set using the label mask INDArray(s) from the
         * DataSet or MultiDataSet. For example, if the network had 2 mask variables called "mask1" and "mask2"
         * and the MultiDataSet label masks should be mapped with {@code MultiDataSet.getLabelMaskArray(0)->"mask1"}
         * and {@code MultiDataSet.getLabelMaskArray(1)->"mask2"}, then this should be set to {@code "mask1", "mask2"}.
         *
         * @param dataSetLabelMaskMapping Name of the variables/placeholders that the feature arrays should be mapped to
         */
        public Builder dataSetLabelMaskMapping(List<String> dataSetLabelMaskMapping){
            this.dataSetLabelMaskMapping = dataSetLabelMaskMapping;
            return this;
        }

        public Builder skipBuilderValidation(boolean skip) {
            this.skipValidation = skip;
            return this;
        }


        private void addEvaluations(boolean validation, @NonNull Map<String, List<IEvaluation>> evaluationMap, @NonNull Map<String, Integer> labelMap,
                @NonNull String variableName, int labelIndex, @NonNull IEvaluation... evaluations) {
            if(evaluationMap.containsKey(variableName) && labelMap.get(variableName) != labelIndex){
                String s;

                if(validation){
                    s = "This ListenerEvaluations.Builder already has validation evaluations for ";
                } else {
                    s = "This ListenerEvaluations.Builder already has train evaluations for ";
                }

                throw new IllegalArgumentException(s + "variable " +
                        variableName + " with label index " + labelIndex + ".  You can't add " +
                        " evaluations with a different label index.  Got label index " + labelIndex);
            }

            if(evaluationMap.containsKey(variableName)){
                evaluationMap.get(variableName).addAll(Arrays.asList(evaluations));
            } else {
                evaluationMap.put(variableName, Arrays.asList(evaluations));
                labelMap.put(variableName, labelIndex);
            }
        }

        /**
         * Add requested History training evaluations for a parm/variable.
         *
         * These evaluations will be reported in the {@link org.nd4j.autodiff.listeners.records.History} object returned by fit.
         *
         * @param variableName  The variable to evaluate
         * @param labelIndex    The index of the label to evaluate against
         * @param evaluations   The evaluations to run
         */
        public Builder trainEvaluation(@NonNull String variableName, int labelIndex, @NonNull IEvaluation... evaluations){
            addEvaluations(false, this.trainEvaluations, this.trainEvaluationLabels, variableName,
                    labelIndex, evaluations);
            return this;
        }

        /**
         * Add requested History training evaluations for a parm/variable.
         *
         * These evaluations will be reported in the {@link org.nd4j.autodiff.listeners.records.History} object returned by fit.
         *
         * @param variable      The variable to evaluate
         * @param labelIndex    The index of the label to evaluate against
         * @param evaluations   The evaluations to run
         */
        public Builder trainEvaluation(@NonNull SDVariable variable, int labelIndex, @NonNull IEvaluation... evaluations){
            return trainEvaluation(variable.name(), labelIndex, evaluations);
        }

        /**
         * Add requested History validation evaluations for a parm/variable.
         *
         * These evaluations will be reported in the {@link org.nd4j.autodiff.listeners.records.History} object returned by fit.
         *
         * @param variableName  The variable to evaluate
         * @param labelIndex    The index of the label to evaluate against
         * @param evaluations   The evaluations to run
         */
        public Builder validationEvaluation(@NonNull String variableName, int labelIndex, @NonNull IEvaluation... evaluations){
            addEvaluations(true, this.validationEvaluations, this.validationEvaluationLabels, variableName,
                    labelIndex, evaluations);
            return this;
        }

        /**
         * Add requested History validation evaluations for a parm/variable.
         *
         * These evaluations will be reported in the {@link org.nd4j.autodiff.listeners.records.History} object returned by fit.
         *
         * @param variable      The variable to evaluate
         * @param labelIndex    The index of the label to evaluate against
         * @param evaluations   The evaluations to run
         */
        public Builder validationEvaluation(@NonNull SDVariable variable, int labelIndex, @NonNull IEvaluation... evaluations){
            return validationEvaluation(variable.name(), labelIndex, evaluations);
        }

        /**
         * Add requested evaluations for a parm/variable, for either training or validation.
         *
         * These evaluations will be reported in the {@link org.nd4j.autodiff.listeners.records.History} object returned by fit.
         *
         * @param validation    Whether to add these evaluations as validation or training
         * @param variableName  The variable to evaluate
         * @param labelIndex    The index of the label to evaluate against
         * @param evaluations   The evaluations to run
         */
        public Builder addEvaluations(boolean validation, @NonNull String variableName, int labelIndex, @NonNull IEvaluation... evaluations){
            if(validation){
                return validationEvaluation(variableName, labelIndex, evaluations);
            } else{
                return trainEvaluation(variableName, labelIndex, evaluations);
            }
        }

        public TrainingConfig build(){
            if(!skipValidation) {
                Preconditions.checkState(updater != null, "Updater (optimizer) must not be null. Use updater(IUpdater) to set an updater");
                Preconditions.checkState(dataSetFeatureMapping != null, "No DataSet feature mapping has been provided. A " +
                        "mapping between DataSet array positions and variables/placeholders must be provided - use dateSetFeatureMapping(...) to set this");
                Preconditions.checkState(markLabelsUnused || dataSetLabelMapping != null, "No DataSet label mapping has been provided. A " +
                        "mapping between DataSet array positions and variables/placeholders must be provided - use dataSetLabelMapping(...) to set this," +
                        " or use markLabelsUnused() to mark labels as unused (for example, for unsupervised learning)");


                Preconditions.checkArgument(trainEvaluations.keySet().equals(trainEvaluationLabels.keySet()),
                        "Must specify a label index for each train evaluation.  Expected: %s, got: %s",
                        trainEvaluations.keySet(), trainEvaluationLabels.keySet());

                Preconditions.checkArgument(validationEvaluations.keySet().equals(validationEvaluationLabels.keySet()),
                        "Must specify a label index for each validation evaluation.  Expected: %s, got: %s",
                        validationEvaluations.keySet(), validationEvaluationLabels.keySet());
            }

            return new TrainingConfig(updater, regularization, minimize, dataSetFeatureMapping, dataSetLabelMapping,
                    dataSetFeatureMaskMapping, dataSetLabelMaskMapping,
                    trainEvaluations, trainEvaluationLabels, validationEvaluations, validationEvaluationLabels, initialLossDataType,
                    computeDataType, masterWeightDataType, lossScaleConfig, gradientAccumulationSteps);
        }
    }


    /**
     * Remove any instances of the specified type from the list.
     * This includes any subtypes.
     * @param list   List. May be null
     * @param remove Type of objects to remove
     */
    public static void removeInstances(List<?> list, Class<?> remove) {
        removeInstancesWithWarning(list, remove, null);
    }

    public static void removeInstancesWithWarning(List<?> list, Class<?> remove, String warning){
        if(list == null || list.isEmpty())
            return;
        Iterator<?> iter = list.iterator();
        while(iter.hasNext()){
            Object o = iter.next();
            if(remove.isAssignableFrom(o.getClass())){
                if(warning != null) {
                    log.warn(warning);
                }
                iter.remove();
            }
        }
    }


    public String toJson(){
        try {
            return JsonMappers.getMapper().writeValueAsString(this);
        } catch (IOException e){
            throw new RuntimeException(e);
        }
    }

    public static TrainingConfig fromJson(@NonNull String json){
        try{
            return JsonMappers.getMapper().readValue(json, TrainingConfig.class);
        } catch (IOException e){
            throw new RuntimeException(e);
        }
    }

    /**
     * Check if per-variable updater configuration is available.
     * Subclasses can override this to indicate they support per-variable settings.
     *
     * @return true if this config supports per-variable updaters
     */
    public boolean hasPerVariableConfig() {
        return false;
    }

    /**
     * Get updater for a specific variable.
     * Default implementation returns the global updater.
     * Subclasses can override this for per-variable support.
     *
     * @param variableName The name of the variable
     * @return The updater to use for this variable
     */
    public IUpdater getUpdaterForVariable(String variableName) {
        return updater;
    }

    /**
     * Get regularization for a specific variable.
     * Default implementation returns the global regularization list.
     * Subclasses can override this for per-variable support.
     *
     * @param variableName The name of the variable
     * @return The regularization list for this variable
     */
    public List<Regularization> getRegularizationForVariable(String variableName) {
        return regularization;
    }

    /**
     * Check if a variable should be trained.
     * Default implementation returns true for all variables.
     * Subclasses can override this to support freezing.
     *
     * @param variableName The name of the variable
     * @return true if the variable should be trained, false if frozen
     */
    public boolean isTrainable(String variableName) {
        return true;
    }
}
