/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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
import org.nd4j.common.base.Preconditions;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.regularization.L1Regularization;
import org.nd4j.linalg.learning.regularization.L2Regularization;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.linalg.learning.regularization.WeightDecay;
import org.nd4j.serde.json.JsonMappers;

import java.io.IOException;
import java.util.*;

/**
 * TrainingConfig is a simple configuration class for defining settings for training a {@link SameDiff} instance.<br>
 * It defines the following settings:<br>
 * <ul>
 *     <li>The {@link IUpdater} to use (i.e., {@link org.nd4j.linalg.learning.config.Adam}, {@link org.nd4j.linalg.learning.config.Nesterovs} etc.
 *     The IUpdater instance is also how the learning rate (or learning rate schedule) is set.</li>
 *     <li>The L1 and L2 regularization coefficients (set to 0.0 by default)</li>
 *     <li>The DataSet feature and label mapping - which defines how the feature/label arrays from the DataSet/MultiDataSet
 *     should be associated with SameDiff variables (usually placeholders)</li>
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
@Slf4j
public class TrainingConfig {

    private IUpdater updater;
    private List<Regularization> regularization = new ArrayList<>();    //Regularization for all trainable parameters
    private boolean minimize = true;
    private List<String> dataSetFeatureMapping;
    private List<String> dataSetLabelMapping;
    private List<String> dataSetFeatureMaskMapping;
    private List<String> dataSetLabelMaskMapping;
    private List<String> lossVariables;
    private int iterationCount;
    private int epochCount;


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
                Collections.<String>emptyList(), Collections.<String>emptyList(), null);
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
                          List<String> dataSetFeatureMaskMapping, List<String> dataSetLabelMaskMapping, List<String> lossVariables) {
        this.updater = updater;
        this.regularization = regularization;
        this.minimize = minimize;
        this.dataSetFeatureMapping = dataSetFeatureMapping;
        this.dataSetLabelMapping = dataSetLabelMapping;
        this.dataSetFeatureMaskMapping = dataSetFeatureMaskMapping;
        this.dataSetLabelMaskMapping = dataSetLabelMaskMapping;
        this.lossVariables = lossVariables;
    }

    protected TrainingConfig(IUpdater updater, List<Regularization> regularization, boolean minimize, List<String> dataSetFeatureMapping, List<String> dataSetLabelMapping,
            List<String> dataSetFeatureMaskMapping, List<String> dataSetLabelMaskMapping, List<String> lossVariables,
            Map<String, List<IEvaluation>> trainEvaluations, Map<String, Integer> trainEvaluationLabels,
            Map<String, List<IEvaluation>> validationEvaluations, Map<String, Integer> validationEvaluationLabels){
        this(updater, regularization, minimize, dataSetFeatureMapping, dataSetLabelMapping, dataSetFeatureMaskMapping, dataSetLabelMaskMapping, lossVariables);
        this.trainEvaluations = trainEvaluations;
        this.trainEvaluationLabels = trainEvaluationLabels;
        this.validationEvaluations = validationEvaluations;
        this.validationEvaluationLabels = validationEvaluationLabels;
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
        private List<Regularization> regularization = new ArrayList<>();
        private boolean minimize = true;
        private List<String> dataSetFeatureMapping;
        private List<String> dataSetLabelMapping;
        private List<String> dataSetFeatureMaskMapping;
        private List<String> dataSetLabelMaskMapping;
        private List<String> lossVariables;
        private boolean skipValidation = false;
        private boolean markLabelsUnused = false;

        private Map<String, List<IEvaluation>> trainEvaluations = new HashMap<>();
        private Map<String, Integer> trainEvaluationLabels = new HashMap<>();

        private Map<String, List<IEvaluation>> validationEvaluations = new HashMap<>();
        private Map<String, Integer> validationEvaluationLabels = new HashMap<>();

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
         * See {@link L1Regularization} for more details
         * @param l1 L1 regularization coefficient
         */
        public Builder l1(double l1){
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

        public Builder skipBuilderValidation(boolean skip){
            this.skipValidation = skip;
            return this;
        }

        public Builder minimize(String... lossVariables){
            this.lossVariables = Arrays.asList(lossVariables);
            return this;
        }

        private void addEvaluations(boolean validation, @NonNull Map<String, List<IEvaluation>> evaluationMap, @NonNull Map<String, Integer> labelMap,
                @NonNull String variableName, int labelIndex, @NonNull IEvaluation... evaluations){
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
                    dataSetFeatureMaskMapping, dataSetLabelMaskMapping, lossVariables,
                    trainEvaluations, trainEvaluationLabels, validationEvaluations, validationEvaluationLabels);
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
}
