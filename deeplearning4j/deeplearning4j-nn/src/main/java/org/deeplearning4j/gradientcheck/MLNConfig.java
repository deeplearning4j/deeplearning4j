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

package org.deeplearning4j.gradientcheck;

import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.Accessors;
import org.nd4j.common.function.Consumer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import java.util.Set;

@Accessors(fluent = true)
@Data
@NoArgsConstructor
public class MLNConfig {
    MultiLayerNetwork net;
    INDArray input;
    INDArray labels;
    INDArray inputMask;
    INDArray labelMask;
    double epsilon = 1e-6;
    double maxRelError = 1e-3;
    double minAbsoluteError = 1e-8;
    PrintMode print = PrintMode.ZEROS;
    boolean exitOnFirstError = false;
    boolean subset;
    int maxPerParam;
    Set<String> excludeParams;
    Consumer<MultiLayerNetwork> callEachIter;
}
