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

package org.deeplearning4j.zoo.util;

import lombok.AllArgsConstructor;
import lombok.Data;

/**
 * ClassPrediction: a prediction for classification, used with a {@link Labels} class.
 * Holds class number, label description, and the prediction probability.
 *
 * @author saudet
 */
@AllArgsConstructor
@Data
public class ClassPrediction {

    private int number;
    private String label;
    private double probability;

    @Override
    public String toString() {
        return "ClassPrediction(number=" + number + ",label=" + label + ",probability=" + probability + ")";
    }
}
