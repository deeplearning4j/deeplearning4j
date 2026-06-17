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

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Helper class for efficient transfer learning with SameDiff.
 * Provides utilities for:
 * <ul>
 *     <li>Pre-computing features from frozen layers</li>
 *     <li>Training only on the unfrozen portion of the network</li>
 *     <li>Caching feature representations for faster training</li>
 * </ul>
 *
 * <p>This is useful when the frozen portion of the network is expensive to compute
 * and you want to train multiple epochs without recomputing the same features.</p>
 *
 * <p>Example usage:</p>
 * <pre>
 * // Create a helper that will freeze everything upstream of "hidden_features"
 * TransferLearningHelper helper = new TransferLearningHelper(model, "hidden_features");
 *
 * // Pre-compute features for all training data
 * List&lt;Map&lt;String, INDArray&gt;&gt; cachedFeatures = new ArrayList&lt;&gt;();
 * while (trainingIterator.hasNext()) {
 *     MultiDataSet batch = trainingIterator.next();
 *     Map&lt;String, INDArray&gt; placeholders = convertToPlaceholders(batch);
 *     cachedFeatures.add(helper.featurize(placeholders));
 * }
 *
 * // Train on cached features (much faster for multiple epochs)
 * for (int epoch = 0; epoch &lt; numEpochs; epoch++) {
 *     for (Map&lt;String, INDArray&gt; features : cachedFeatures) {
 *         helper.fitOnFeatures(features);
 *     }
 * }
 * </pre>
 *
 * @author Adam Gibson
 * @see TransferLearning
 */
@Slf4j
public class TransferLearningHelper {

    @Getter
    private final SameDiff sameDiff;

    @Getter
    private final List<String> featureOutputNames;

    @Getter
    private final Set<String> frozenVariables;

    /**
     * Create a helper for the given model with specified feature extraction points.
     * All trainable variables upstream of the feature outputs will be frozen.
     *
     * @param sameDiff       The SameDiff model
     * @param featureOutputs Names of variables to use as feature outputs
     */
    public TransferLearningHelper(SameDiff sameDiff, String... featureOutputs) {
        this.sameDiff = sameDiff;
        this.featureOutputNames = Arrays.asList(featureOutputs);

        // Find and freeze upstream variables
        if (featureOutputs.length > 0) {
            this.frozenVariables = TransferLearning.Builder.findUpstreamVariables(sameDiff, featureOutputs);
            sameDiff.freezeVariables(frozenVariables);
        } else {
            // No feature outputs specified - collect existing frozen variables
            this.frozenVariables = collectFrozenVariables(sameDiff);
        }
    }

    /**
     * Create a helper for an already-configured model.
     * This collects existing frozen variables without modifying the model.
     *
     * @param sameDiff The SameDiff model (already configured for transfer learning)
     */
    public TransferLearningHelper(SameDiff sameDiff) {
        this.sameDiff = sameDiff;
        this.featureOutputNames = Collections.emptyList();
        this.frozenVariables = collectFrozenVariables(sameDiff);
    }

    /**
     * Collect all frozen (CONSTANT type) variables that were originally trainable.
     * Note: This only detects variables that are CONSTANT type. To properly identify
     * frozen variables, use the constructor with feature outputs or track freezing explicitly.
     */
    private Set<String> collectFrozenVariables(SameDiff sd) {
        // Check for CONSTANT type - frozen variables are converted from VARIABLE to CONSTANT
        return sd.variables().stream()
                .filter(var -> var.getVariableType() == VariableType.CONSTANT)
                .map(SDVariable::name)
                .collect(Collectors.toCollection(HashSet::new));
    }

    /**
     * Compute features for the given input placeholders.
     * This runs the frozen portion of the network and returns the feature outputs.
     *
     * @param placeholders Input placeholder values
     * @return Map of feature output name to computed feature array
     */
    public Map<String, INDArray> featurize(Map<String, INDArray> placeholders) {
        if (featureOutputNames.isEmpty()) {
            log.warn("No feature outputs specified. Returning empty map.");
            return Collections.emptyMap();
        }

        Map<String, INDArray> result = sameDiff.output(placeholders, featureOutputNames);
        return result;
    }

    /**
     * Get the number of trainable (unfrozen) variables.
     *
     * @return Count of trainable variables
     */
    public int getTrainableVariableCount() {
        return (int) sameDiff.variables().stream()
                .filter(var -> var.getVariableType() == VariableType.VARIABLE)
                .count();
    }

    /**
     * Get the number of frozen variables.
     *
     * @return Count of frozen variables
     */
    public int getFrozenVariableCount() {
        return frozenVariables.size();
    }

    /**
     * Get the names of all trainable (unfrozen) variables.
     *
     * @return Set of trainable variable names
     */
    public Set<String> getTrainableVariables() {
        return sameDiff.variables().stream()
                .filter(var -> var.getVariableType() == VariableType.VARIABLE)
                .map(SDVariable::name)
                .collect(Collectors.toCollection(HashSet::new));
    }

    /**
     * Train the unfrozen portion of the network on pre-computed features.
     * This assumes the features correspond to the feature extraction points.
     *
     * @param featurizedIterator Iterator over pre-computed features
     * @param epochs             Number of epochs to train
     * @return Training history
     */
    public History fitFeaturized(MultiDataSetIterator featurizedIterator, int epochs) {
        return sameDiff.fit(featurizedIterator, epochs);
    }

    /**
     * Print a summary of frozen vs trainable parameters.
     */
    public void printSummary() {
        long trainableParams = sameDiff.variables().stream()
                .filter(var -> var.getArr() != null && var.getVariableType() == VariableType.VARIABLE)
                .mapToLong(var -> var.getArr().length())
                .sum();

        long frozenParams = sameDiff.variables().stream()
                .filter(var -> var.getArr() != null && frozenVariables.contains(var.name()))
                .mapToLong(var -> var.getArr().length())
                .sum();

        log.info("Transfer Learning Summary:");
        log.info("  Frozen variables: {} ({} parameters)", frozenVariables.size(), frozenParams);
        log.info("  Trainable variables: {} ({} parameters)", getTrainableVariableCount(), trainableParams);
        log.info("  Feature outputs: {}", featureOutputNames);
    }
}
