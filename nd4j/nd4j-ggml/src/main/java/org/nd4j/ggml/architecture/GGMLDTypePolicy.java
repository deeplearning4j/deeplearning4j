/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.ggml.architecture;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.buffer.DataType;

/**
 * Numeric precision policy shared by GGML model importers.
 *
 * <p>GGUF tensors retain their model storage type, while reductions, matrix
 * accumulation and recurrent state updates use FP32 for low-precision model
 * types. Keeping those transitions here prevents individual architectures from
 * embedding their own dtype tables and unconditional casts.</p>
 */
final class GGMLDTypePolicy {

    private GGMLDTypePolicy() {
    }

    static boolean requiresFp32Accumulation(DataType dataType) {
        return dataType == DataType.HALF
                || dataType == DataType.BFLOAT16
                || dataType == DataType.FLOAT8
                || dataType == DataType.FLOAT8_E5M2;
    }

    static DataType accumulationType(DataType storageType) {
        return requiresFp32Accumulation(storageType) ? DataType.FLOAT : storageType;
    }

    static SDVariable castForAccumulation(SDVariable variable, String name) {
        return castTo(variable, name, accumulationType(variable.dataType()));
    }

    static SDVariable castTo(SDVariable variable, String name, DataType targetType) {
        return variable.dataType() == targetType ? variable : variable.castTo(name, targetType);
    }
}
