/*
 *  ******************************************************************************
 *  *
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

package org.deeplearning4j.nn.weights.embeddings;

import lombok.EqualsAndHashCode;
import lombok.NonNull;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.nd4j.common.base.Preconditions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonIgnoreProperties;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * Weight initialization for initializing the parameters of an EmbeddingLayer from a {@link EmbeddingInitializer}
 *
 * Note: WeightInitEmbedding supports both JSON serializable and non JSON serializable initializations.
 * In the case of non-JSON serializable embeddings, they are a one-time only use: once they have been used
 * to initialize the parameters, they will be removed from the WeightInitEmbedding instance.
 * This is to prevent unnecessary references to potentially large objects in memory (i.e., to avoid memory leaks)
 *
 * @author Alex Black
 */
@JsonIgnoreProperties("nonSerializableInit")
@EqualsAndHashCode
public class WeightInitEmbedding implements IWeightInit {

    private EmbeddingInitializer serializableInit;
    private EmbeddingInitializer nonSerializableInit;

    public WeightInitEmbedding(@NonNull EmbeddingInitializer embeddingInitializer){
        this((embeddingInitializer.jsonSerializable() ? embeddingInitializer : null), (embeddingInitializer.jsonSerializable() ? null : embeddingInitializer));

    }

    protected WeightInitEmbedding(@JsonProperty("serializableInit") EmbeddingInitializer serializableInit,
                                  @JsonProperty("nonSerializableInit") EmbeddingInitializer nonSerializableInit){
        this.serializableInit = serializableInit;
        this.nonSerializableInit = nonSerializableInit;
    }

    @Override
    public INDArray init(double fanIn, double fanOut, long[] shape, char order, INDArray paramView) {
        EmbeddingInitializer init = serializableInit != null ? serializableInit : nonSerializableInit;
        if(init == null){
            throw new IllegalStateException("Cannot initialize embedding layer weights: no EmbeddingInitializer is available." +
                    " This can occur if you save network configuration, load it, and the try to ");
        }

        Preconditions.checkState(shape[0] == init.vocabSize(), "Parameters shape[0]=%s does not match embedding initializer vocab size of %s", shape[0], init.vocabSize());
        Preconditions.checkState(shape[1] == init.vectorSize(), "Parameters shape[1]=%s does not match embedding initializer vector size of %s", shape[1], init.vectorSize());

        INDArray reshaped = paramView.reshape('c', shape);
        init.loadWeightsInto(reshaped);

        //Now that we've loaded weights - let's clear the reference if it's non-serializable so it can be GC'd
        this.nonSerializableInit = null;

        return reshaped;
    }

    public long[] shape(){
        if(serializableInit != null){
            return new long[]{serializableInit.vocabSize(), serializableInit.vectorSize()};
        } else if(nonSerializableInit != null){
            return new long[]{nonSerializableInit.vocabSize(), nonSerializableInit.vectorSize()};
        }
        return null;
    }
}
