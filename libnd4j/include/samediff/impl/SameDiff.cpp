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

#include "../SameDiff.h"

namespace samediff {
    SDVariable SameDiff::variable(const char *name, const nd4j::NDArray &array) {
        return SDVariable();
    }

    SDVariable SameDiff::placeholder(const char *name, const nd4j::DataType dataType, const std::vector<Nd4jLong> shape) {
        return SDVariable();
    }

    void SameDiff::execute() {
        //
    }

    void SameDiff::train() {
        //
    }

    void SameDiff::executeWithDictionary(const std::unordered_map<const char*, nd4j::NDArray> &args) {
        //
    }

    void SameDiff::save(const char *filename) {
        //
    }

    SameDiff SameDiff::load(const char *filename) {
        return {};
    }
}