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

package org.deeplearning4j.nn.api;

import org.nd4j.adapters.OutputAdapter;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface ModelAdapter<T> extends OutputAdapter<T> {
    /**
     * This method invokes model internally, and does convertion to T
     * @return
     */
    T apply(Model model, INDArray[] inputs, INDArray[] inputMasks, INDArray[] labelsMasks);
}
