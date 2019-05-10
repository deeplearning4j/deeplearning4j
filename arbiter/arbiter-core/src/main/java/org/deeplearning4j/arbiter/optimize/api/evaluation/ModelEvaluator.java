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

package org.deeplearning4j.arbiter.optimize.api.evaluation;

import org.deeplearning4j.arbiter.optimize.api.data.DataProvider;

import java.io.Serializable;
import java.util.List;

/**
 * ModelEvaluator: Used to conduct additional evaluation.
 * For example, this may be classification performance on a test set or similar
 */
public interface ModelEvaluator extends Serializable {
    Object evaluateModel(Object model, DataProvider dataProvider);

    /**
     * @return The model types supported by this class
     */
    List<Class<?>> getSupportedModelTypes();

    /**
     * @return The datatypes supported by this class
     */
    List<Class<?>> getSupportedDataTypes();
}
