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

package org.nd4j.parameterserver.distributed.v2.util;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.Serializable;

/**
 * This class provides simple holder functionaliy
 *
 * @author raver119@gmail.com
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class UpdaterParametersHolder implements Serializable {
    private INDArray parameters;

    private long timeReceived = System.currentTimeMillis();

    private boolean finalState = false;

    public long getAge() {
        return System.currentTimeMillis()  - timeReceived;
    }
}
