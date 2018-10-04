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
#include <ops/BroadcastOpsTuple.h>

namespace nd4j {
    BroadcastOpsTuple BroadcastOpsTuple::custom(nd4j::scalar::Ops scalar, nd4j::pairwise::Ops pairwise, nd4j::broadcast::Ops broadcast) {
        BroadcastOpsTuple t(scalar, pairwise, broadcast);
        return t;
    }

    BroadcastOpsTuple BroadcastOpsTuple::Add() {
        return custom(nd4j::scalar::Add, nd4j::pairwise::Add, nd4j::broadcast::Add);
    }

    BroadcastOpsTuple BroadcastOpsTuple::Assign() {
        return custom(nd4j::scalar::Copy, nd4j::pairwise::Copy, nd4j::broadcast::Copy);
    }

    BroadcastOpsTuple BroadcastOpsTuple::Divide() {
        return custom(nd4j::scalar::Divide, nd4j::pairwise::Divide, nd4j::broadcast::Divide);
    }

    BroadcastOpsTuple BroadcastOpsTuple::Multiply() {
        return custom(nd4j::scalar::Multiply, nd4j::pairwise::Multiply, nd4j::broadcast::Multiply);
    }

    BroadcastOpsTuple BroadcastOpsTuple::Subtract() {
        return custom(nd4j::scalar::Subtract, nd4j::pairwise::Subtract, nd4j::broadcast::Subtract);
    }
}
