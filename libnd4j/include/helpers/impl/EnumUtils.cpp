/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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

#include <graph/VariableType.h>
#include <helpers/EnumUtils.h>

using namespace nd4j::graph;

namespace nd4j {
    const char * EnumUtils::_VariableTypeToString(nd4j::graph::VariableType variableType) {
        switch (variableType) {
            case NDARRAY: return "NDARRAY";
            case ARRAY_LIST: return "ARRAY_LIST";
            case FLOW: return "FLOW";
            default: return "UNKNOWN VariableType";
        }
    }

    const char * EnumUtils::_OpTypeToString(nd4j::graph::OpType opType) {
        switch(opType) {
            case OpType_REDUCE_SAME: return "REDUCE_SAME";
            case OpType_REDUCE_BOOL: return "REDUCE_BOOL";
            case OpType_REDUCE_LONG: return "REDUCE_LONG";
            case OpType_REDUCE_FLOAT: return "REDUCE_FLOAT";
            case OpType_BOOLEAN: return "BOOLEAN";
            case OpType_BROADCAST: return "BROADCAST";
            case OpType_BROADCAST_BOOL: return "BROADCAST_BOOL";
            case OpType_PAIRWISE: return "PAIRWISE";
            case OpType_PAIRWISE_BOOL: return "PAIRWISE_BOOL";
            case OpType_CUSTOM: return "CUSTOM";
            case OpType_LOGIC: return "LOGIC";
            case OpType_TRANSFORM_SAME: return "TRANSFORM_SAME";
            case OpType_TRANSFORM_FLOAT: return "TRANSFORM_FLOAT";
            case OpType_TRANSFORM_BOOL: return "TRANSFORM_BOOL";
            case OpType_TRANSFORM_STRICT: return "TRANSFORM_STRICT";
            case OpType_TRANSFORM_ANY: return "TRANSFORM_ANY";
            case OpType_INDEX_REDUCE: return "INDEX_ACCUMULATION";
            case OpType_SCALAR: return "SCALAR";
            case OpType_SCALAR_BOOL: return "SCALAR_BOOL";
            case OpType_SHAPE: return "SHAPE";
            default: return "UNKNOWN OpType";
        }
    }


    const char * EnumUtils::_LogicOpToString(int opNum) {
        switch(opNum) {
            case 0: return "WHILE";
            case 10: return "SCOPE";
            case 20: return "CONDITIONAL";
            case 30: return "SWITCH";
            case 40: return "RETURN";
            case 60: return "MERGE";
            case 70: return "LOOP_COND";
            case 80: return "NEXT_ITERATION";
            case 90: return "EXIT";
            case 100: return "ENTER";
            default: return "UNKNOWN OPERATION";
        } 
    }
}