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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 09.04.2019
//

#ifndef DEV_TESTS_BROADCASTPAIRWISECONVERTER_H
#define DEV_TESTS_BROADCASTPAIRWISECONVERTER_H

#include <op_boilerplate.h>
#include <stdexcept>

namespace nd4j {

//////////////////////////////////////////////////////////////////////////
inline pairwise::Ops fromBroadcastToPairwise(broadcast::Ops op) {
    switch (op) {
        case broadcast::Add: return pairwise::Add;
        case broadcast::Subtract: return pairwise::Subtract;
        case broadcast::Multiply: return pairwise::Multiply;
        case broadcast::Divide: return pairwise::Divide;
        case broadcast::ReverseDivide: return pairwise::ReverseDivide;
        case broadcast::ReverseSubtract: return pairwise::ReverseSubtract;
        case broadcast::CopyPws: return pairwise::CopyPws;
        case broadcast::Pow: return pairwise::Pow;
        case broadcast::MinPairwise: return pairwise::MinPairwise;
        case broadcast::MaxPairwise: return pairwise::MaxPairwise;
        case broadcast::AMinPairwise: return pairwise::AMinPairwise;
        case broadcast::AMaxPairwise: return pairwise::AMaxPairwise;
        case broadcast::SquaredSubtract: return pairwise::SquaredSubtract;
        case broadcast::FloorMod: return pairwise::FloorMod;
        case broadcast::FloorDiv: return pairwise::FloorDiv;
        case broadcast::ReverseMod: return pairwise::ReverseMod;
        case broadcast::SafeDivide: return pairwise::SafeDivide;
        case broadcast::Mod: return pairwise::Mod;
        case broadcast::TruncateDiv: return pairwise::TruncateDiv;
        case broadcast::Atan2: return pairwise::Atan2;
        case broadcast::LogicalOr: return pairwise::LogicalOr;
        case broadcast::LogicalXor: return pairwise::LogicalXor;
        case broadcast::LogicalNot: return pairwise::LogicalNot;
        case broadcast::LogicalAnd: return pairwise::LogicalAnd;
        default:
            throw std::runtime_error("fromBroadcastToPairwise: Not convertible operation");
    }
}

//////////////////////////////////////////////////////////////////////////
inline pairwise::BoolOps fromBroadcastToPairwiseBool(broadcast::BoolOps op) {
    switch (op) {
        case broadcast::EqualTo: return pairwise::EqualTo;
        case broadcast::GreaterThan: return pairwise::GreaterThan;
        case broadcast::LessThan: return pairwise::LessThan;
        case broadcast::Epsilon: return pairwise::Epsilon;
        case broadcast::GreaterThanOrEqual: return pairwise::GreaterThanOrEqual;
        case broadcast::LessThanOrEqual: return pairwise::LessThanOrEqual;
        case broadcast::NotEqualTo: return pairwise::NotEqualTo;
        case broadcast::And: return pairwise::And;
        case broadcast::Or: return pairwise::Or;
        case broadcast::Xor: return pairwise::Xor;
        case broadcast::Not: return pairwise::Not;        
        default:
            throw std::runtime_error("fromBroadcastToPairwiseBool: Not convertible operation");
    }
}

}

#endif //DEV_TESTS_BROADCASTPAIRWISECONVERTER_H