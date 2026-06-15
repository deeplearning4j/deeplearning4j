/*
 *  ******************************************************************************
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

package org.eclipse.deeplearning4j.ggml;

/**
 * GGUF metadata value types.
 */
public enum GGUFMetadataType {
    UINT8(0),
    INT8(1),
    UINT16(2),
    INT16(3),
    UINT32(4),
    INT32(5),
    FLOAT32(6),
    BOOL(7),
    STRING(8),
    ARRAY(9),
    UINT64(10),
    INT64(11),
    FLOAT64(12),
    UNKNOWN(-1);

    private final int id;

    GGUFMetadataType(int id) {
        this.id = id;
    }

    public int getId() {
        return id;
    }

    public static GGUFMetadataType fromId(int id) {
        for (GGUFMetadataType type : values()) {
            if (type.id == id) {
                return type;
            }
        }
        return UNKNOWN;
    }
}
