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

package org.deeplearning4j.nn.updater;

import lombok.AllArgsConstructor;
import lombok.Data;
import org.deeplearning4j.nn.api.Trainable;
import org.nd4j.linalg.api.ndarray.INDArray;

@AllArgsConstructor
@Data
public class ParamState {
    private final Trainable layer;
    private final String paramName;
    private final int paramOffsetStart;
    private final int paramOffsetEnd;
    private final INDArray paramView;
    private final INDArray gradView;
}
