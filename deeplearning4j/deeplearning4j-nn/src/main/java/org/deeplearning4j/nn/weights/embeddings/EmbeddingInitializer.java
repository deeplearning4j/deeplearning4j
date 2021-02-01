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

package org.deeplearning4j.nn.weights.embeddings;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * An interface implemented by things like Word2Vec etc that allows them to be used as weight
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface EmbeddingInitializer extends Serializable {

    /**
     * Load the weights into the specified INDArray
     * @param array Array of shape [vocabSize, vectorSize]
     */
    void loadWeightsInto(INDArray array);

    /**
     * @return Size of the vocabulary
     */
    long vocabSize();

    /**
     * @return Size of each vector
     */
    int vectorSize();

    /**
     * @return True if the embedding initializer can be safely serialized as JSON
     */
    boolean jsonSerializable();
}
