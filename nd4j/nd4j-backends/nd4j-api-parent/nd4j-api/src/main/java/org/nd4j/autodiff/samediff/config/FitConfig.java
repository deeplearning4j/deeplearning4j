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

package org.nd4j.autodiff.samediff.config;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.dataset.adapter.MultiDataSetIteratorAdapter;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

/**
 * Configuration for a {@link SameDiff} training operation.
 * <p>
 * Used in {@link SameDiff#fit()}.
 */
@Getter
@Setter
public class FitConfig {

    @Setter(AccessLevel.NONE)
    private SameDiff sd;

    private MultiDataSetIterator trainingData;

    private MultiDataSetIterator validationData = null;

    private int epochs = -1;

    private int validationFrequency = 1;

    @NonNull
    private List<Listener> listeners = new ArrayList<>();

    public FitConfig(@NonNull SameDiff sd) {
        this.sd = sd;
    }

    /**
     * Set the number of epochs to train for
     */
    public FitConfig epochs(int epochs) {
        this.epochs = epochs;
        return this;
    }

    /**
     * Set the training data
     */
    public FitConfig train(@NonNull MultiDataSetIterator trainingData) {
        this.trainingData = trainingData;
        return this;
    }

    /**
     * Set the training data
     */
    public FitConfig train(@NonNull DataSetIterator trainingData) {
        return train(new MultiDataSetIteratorAdapter(trainingData));
    }

    /**
     * Set the training data and number of epochs
     */
    public FitConfig train(@NonNull MultiDataSetIterator trainingData, int epochs) {
        return train(trainingData).epochs(epochs);
    }

    /**
     * Set the training data and number of epochs
     */
    public FitConfig train(@NonNull DataSetIterator trainingData, int epochs) {
        return train(trainingData).epochs(epochs);
    }

    /**
     * Set the validation data
     */
    public FitConfig validate(MultiDataSetIterator validationData) {
        this.validationData = validationData;
        return this;
    }

    /**
     * Set the validation data
     */
    public FitConfig validate(DataSetIterator validationData) {
        if (validationData == null) {
            return validate((MultiDataSetIterator) null);
        } else {
            return validate(new MultiDataSetIteratorAdapter(validationData));
        }
    }

    /**
     * Set the validation frequency.  Validation will be preformed once every so many epochs.
     * <p>
     * Specifically, validation will be preformed when i % validationFrequency == 0
     */
    public FitConfig validationFrequency(int validationFrequency) {
        this.validationFrequency = validationFrequency;
        return this;
    }

    /**
     * Set the validation data and frequency
     * <p>
     * Specifically, validation will be preformed when i % validationFrequency == 0
     */
    public FitConfig validate(MultiDataSetIterator validationData, int validationFrequency) {
        return validate(validationData).validationFrequency(validationFrequency);
    }

    /**
     * Set the validation data and frequency
     * <p>
     * Specifically, validation will be preformed when i % validationFrequency == 0
     */
    public FitConfig validate(DataSetIterator validationData, int validationFrequency) {
        return validate(validationData).validationFrequency(validationFrequency);
    }

    /**
     * Add listeners for this operation
     */
    public FitConfig listeners(@NonNull Listener... listeners) {
        this.listeners.addAll(Arrays.asList(listeners));
        return this;
    }


    private void validateConfig() {
        Preconditions.checkNotNull(trainingData, "Training data must not be null");
        Preconditions.checkState(epochs > 0, "Epochs must be > 0, got %s", epochs);

        if (validationData != null)
            Preconditions.checkState(validationFrequency > 0, "Validation Frequency must be > 0 if validation data is given, got %s", validationFrequency);
    }

    /**
     * Do the training.
     *
     * @return a {@link History} object containing the history information for this training operation
     * (evaluations specified in the {@link TrainingConfig}, loss values, and timing information).
     */
    public History exec() {
        validateConfig();

        return sd.fit(trainingData, epochs, validationData, validationFrequency, listeners.toArray(new Listener[0]));
    }

}
