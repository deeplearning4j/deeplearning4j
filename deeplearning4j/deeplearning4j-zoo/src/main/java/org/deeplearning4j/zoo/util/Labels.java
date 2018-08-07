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

import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;

/**
 * Interface to helper classes that return label descriptions.
 *
 * @author saudet
 */
public interface Labels {

    /**
     * Returns the description of the nth class from the classes of a dataset.
     * @param n
     * @return label description
     */
    String getLabel(int n);

    /**
     * Given predictions from the trained model this method will return a list
     * of the top n matches and the respective probabilities.
     * @param predictions raw
     * @return decoded predictions
     */
    List<List<ClassPrediction>> decodePredictions(INDArray predictions, int n);
}
