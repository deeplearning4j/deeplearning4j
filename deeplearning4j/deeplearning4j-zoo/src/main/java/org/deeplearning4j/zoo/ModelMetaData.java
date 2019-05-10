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

package org.deeplearning4j.zoo;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * Metadata describing a model, including input shapes. This is helpful for instantiating
 * the model programmatically and ensuring appropriate inputs are used.
 *
 * @deprecated As of May 10, 2018. Will be removed in v1.1. Getters are now available on the inputShape from the ZooModel implementation.
 */
@Getter
@AllArgsConstructor
@Deprecated
public class ModelMetaData {
    private int[][] inputShape;
    private int numOutputs;
    private ZooType zooType;

    /**
     * If number of inputs are greater than 1, this states that the
     * implementation should use MultiDataSet.
     * @return
     */
    public boolean useMDS() {
        return inputShape.length > 1 ? true : false;
    }
}
