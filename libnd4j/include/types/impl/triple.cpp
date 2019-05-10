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
// Created by raver on 4/5/2018.
//

#include <types/triple.h>

namespace nd4j {
    int Triple::first() const {
        return _first;
    }

    int Triple::second() const {
        return _second;
    }

    int Triple::third() const {
        return _third;
    }

    Triple::Triple(int first, int second, int third) {
        _first = first;
        _second = second;
        _third = third;
    }
}
