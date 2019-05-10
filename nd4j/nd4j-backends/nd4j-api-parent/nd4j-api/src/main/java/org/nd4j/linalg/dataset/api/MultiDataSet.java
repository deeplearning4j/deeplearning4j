/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.nd4j.linalg.dataset.api;

import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.*;
import java.util.List;

/**
 * MultiDataSet is an interface for representing complex data sets, that have (potentially) multiple inputs and outputs
 * For example, some complex neural network architectures may have multiple independent inputs, and multiple independent
 * outputs. These inputs and outputs need not even be the same opType of data: for example, images in and sequences out, etc
 */
public interface MultiDataSet extends Serializable {

    /**
     * Number of arrays of features/input data in the MultiDataSet
     */
    int numFeatureArrays();

    /**
     * Number of arrays of label/output data in the MultiDataSet
     */
    int numLabelsArrays();

    /**
     * Get all of the input features, as an array of INDArrays
     */
    INDArray[] getFeatures();

    /**
     * Get a single feature/input array
     */
    INDArray getFeatures(int index);

    /**
     * Set all of the features arrays for the MultiDataSet
     */
    void setFeatures(INDArray[] features);

    /**
     * Set a single features array (by index) for the MultiDataSet
     */
    void setFeatures(int idx, INDArray features);

    /**
     * Get all of the labels, as an array of INDArrays
     */
    INDArray[] getLabels();

    /**
     * Get a single label/output array
     */
    INDArray getLabels(int index);

    /**
     * Set all of the labels arrays for the MultiDataSet
     */
    void setLabels(INDArray[] labels);

    /**
     * Set a single labels array (by index) for the MultiDataSet
     */
    void setLabels(int idx, INDArray labels);

    /**
     * Whether there are any mask arrays (features or labels) present for this MultiDataSet
     */
    boolean hasMaskArrays();

    /**
     * Get the feature mask arrays. May return null if no feature mask arrays are present; otherwise,
     * any entry in the array may be null if no mask array is present for that particular feature
     */
    INDArray[] getFeaturesMaskArrays();

    /**
     * Get the specified feature mask array. Returns null if no feature mask is present for that particular feature/input
     */
    INDArray getFeaturesMaskArray(int index);

    /**
     * Set the feature mask arrays
     */
    void setFeaturesMaskArrays(INDArray[] maskArrays);

    /**
     * Set a single feature mask array by index
     */
    void setFeaturesMaskArray(int idx, INDArray maskArray);

    /**
     * Get the labels mask arrays. May return null if no labels mask arrays are present; otherwise,
     * any entry in the array may be null if no mask array is present for that particular label
     */
    INDArray[] getLabelsMaskArrays();

    /**
     * Get the specified label mask array. Returns null if no label mask is present for that particular feature/input
     */
    INDArray getLabelsMaskArray(int index);

    /**
     * Set the labels mask arrays
     */
    void setLabelsMaskArray(INDArray[] labels);

    /**
     * Set a single labels mask array by index
     */
    void setLabelsMaskArray(int idx, INDArray labelsMaskArray);

    /**
     * Save this MultiDataSet to the specified stream. Stream will be closed after saving.
     */
    void save(OutputStream to) throws IOException;

    /**
     * Save this MultiDataSet to the specified file
     */
    void save(File to) throws IOException;

    /**
     * Load the contents of this MultiDataSet from the specified stream. Stream will be closed after loading.
     */
    void load(InputStream from) throws IOException;

    /**
     * Load the contents of this MultiDataSet from the specified file
     */
    void load(File from) throws IOException;

    /**
     * SplitV the MultiDataSet into a list of individual examples.
     *
     * @return List of MultiDataSets, each with 1 example
     */
    List<MultiDataSet> asList();

    /**
     * Clone the dataset
     *
     * @return a clone of the dataset
     */
    MultiDataSet copy();

    /**
     * Set the metadata for this MultiDataSet<br>
     * By convention: the metadata can be any serializable object, one per example in the MultiDataSet
     *
     * @param exampleMetaData Example metadata to set
     */
    void setExampleMetaData(List<? extends Serializable> exampleMetaData);

    /**
     * Get the example metadata, or null if no metadata has been set<br>
     * Note: this method results in an unchecked cast - care should be taken when using this!
     *
     * @param metaDataType Class of the metadata (used for opType information)
     * @param <T>          Type of metadata
     * @return List of metadata objects
     */
    <T extends Serializable> List<T> getExampleMetaData(Class<T> metaDataType);

    /**
     * Get the example metadata, or null if no metadata has been set
     *
     * @return List of metadata instances
     * @see {@link #getExampleMetaData(Class)} for convenience method for types
     */
    List<Serializable> getExampleMetaData();

    /**
     * This method returns memory amount occupied by this MultiDataSet.
     *
     * @return value in bytes
     */
    long getMemoryFootprint();

    /**
     * This method migrates this MultiDataSet into current Workspace (if any)
     */
    void migrate();

    /**
     * This method detaches this MultiDataSet from current Workspace (if any)
     */
    void detach();

    /**
     * @return True if the MultiDataSet is empty (all features/labels arrays are empty)
     */
    boolean isEmpty();

    /**
     * Shuffle the order of the examples in the MultiDataSet. Note that this generally won't make any difference in
     * practice unless the MultiDataSet is later split.
     */
    void shuffle();
}
