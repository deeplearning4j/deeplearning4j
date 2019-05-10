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

package org.deeplearning4j.datasets.iterator;

import org.nd4j.linalg.factory.Nd4j;

/**
 * RandomDataSetIterator: Generates random values (or zeros, ones, integers, etc) according to some distribution.<br>
 * The type of values produced can be specified by the {@link Values} enumeration.<br>
 * Note: This is typically used for testing, debugging and benchmarking purposes.
 *
 * @author Alex Black
 */
public class RandomDataSetIterator extends MultiDataSetWrapperIterator {

    public enum Values {RANDOM_UNIFORM, RANDOM_NORMAL, ONE_HOT, ZEROS, ONES, BINARY, INTEGER_0_10, INTEGER_0_100, INTEGER_0_1000,
        INTEGER_0_10000, INTEGER_0_100000;
        public RandomMultiDataSetIterator.Values toMdsValues(){
            return RandomMultiDataSetIterator.Values.valueOf(this.toString());
        }
    };

    /**
     * @param numMiniBatches Number of minibatches per epoch
     * @param featuresShape  Features shape
     * @param labelsShape    Labels shape
     * @param featureValues  Type of values for the features
     * @param labelValues    Type of values for the labels
     */
    public RandomDataSetIterator(int numMiniBatches, long[] featuresShape, long[] labelsShape, Values featureValues, Values labelValues){
        this(numMiniBatches, featuresShape, labelsShape, featureValues, labelValues, Nd4j.order(), Nd4j.order());
    }

    /**
     * @param numMiniBatches Number of minibatches per epoch
     * @param featuresShape  Features shape
     * @param labelsShape    Labels shape
     * @param featureValues  Type of values for the features
     * @param labelValues    Type of values for the labels
     * @param featuresOrder  Array order ('c' or 'f') for the features array
     * @param labelsOrder    Array order ('c' or 'f') for the labels array
     */
    public RandomDataSetIterator(int numMiniBatches, long[] featuresShape, long[] labelsShape, Values featureValues, Values labelValues,
                                 char featuresOrder, char labelsOrder){
        super(new RandomMultiDataSetIterator.Builder(numMiniBatches)
                .addFeatures(featuresShape, featuresOrder, featureValues.toMdsValues())
                .addLabels(labelsShape, labelsOrder, labelValues.toMdsValues())
        .build());
    }

}
