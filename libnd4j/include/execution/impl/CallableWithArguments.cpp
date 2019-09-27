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

#include <execution/CallableWithArguments.h>

namespace samediff {
    CallableWithArguments::CallableWithArguments(std::function<void(uint64_t, uint64_t, uint64_t)> &func, uint64_t start_x,
                                                 uint64_t stop_x, uint64_t increment_x) {
        _function_1d = func;
        _arguments = {start_x, stop_x, increment_x};
        _finished = false;
    }

    CallableWithArguments::CallableWithArguments(
            std::function<void(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)> &func, uint64_t start_x,
            uint64_t stop_x, uint64_t increment_x, uint64_t start_y, uint64_t stop_y, uint64_t increment_y) {
        _function_2d = func;
        _arguments = {start_x, stop_x, increment_x, start_y, stop_y, increment_y};
        _finished = false;
    }

    int CallableWithArguments::dimensions() {
        return _arguments.size() / 3;
    }

    std::vector<uint64_t>& CallableWithArguments::arguments() {
        return _arguments;
    }

    bool CallableWithArguments::finished() {
        return _finished.load();
    }

    void CallableWithArguments::finish() {
        _finished = true;
        _condition.notify_one();
    }

    void CallableWithArguments::waitUntilFinished() {
        std::unique_lock<std::mutex> lock(_lock);
        _condition.wait(lock, [=]{ return _finished.load(); });
    }


    std::function<void(uint64_t, uint64_t, uint64_t)> CallableWithArguments::function_1d() {
        return _function_1d;
    }

    std::function<void(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t)> CallableWithArguments::function_2d() {
        return _function_2d;
    }
}