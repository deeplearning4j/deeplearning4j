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

//
// @author raver119@gmail.com
//

#ifndef ND4J_DATATYPE_H
#define ND4J_DATATYPE_H

namespace nd4j {
    enum DataType {
        INHERIT = 0,
        BOOL = 1,
        FLOAT8 = 2,
        HALF = 3,
        HALF2 = 4,
        FLOAT = 5,
        DOUBLE = 6,
        INT8 = 7,
        INT16 = 8,
        INT32 = 9,
        INT64 = 10,
        UINT8 = 11,
        UINT16 = 12,
        UINT32 = 13,
        UINT64 = 14,
        QINT8 = 15,
        QINT16 = 16,
    };
}

#endif