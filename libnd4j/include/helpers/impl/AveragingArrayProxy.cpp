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

#include "../AveragingArrayProxy.h"

namespace nd4j {
    AveragingArrayProxy::AveragingArrayProxy(NDArray *original) {
        _original = original;
    }

    AveragingArrayProxy::~AveragingArrayProxy() {
        for (auto v:_references)
            delete v;
    }

    bool AveragingArrayProxy::writeableExists(std::pair<int, int> &key) {
        _lock.lock();

        auto r = _writeables.count(key) > 0;

        _lock.unlock();

        return r;
    }

    bool AveragingArrayProxy::writeableExists(int row, int key) {
        std::pair<int, int> k(row, key);
        return writeableExists(k);
    }

    NDArray* AveragingArrayProxy::readable(int row, int key) {
        std::pair<int, int> k(row, key);

        if (writeableExists(k)) {
            _lock.lock();

            auto r = _writeables[k];

            _lock.unlock();

            return r;
        } else {
            auto readable = (*_original)({row,row+1, 0,0});

            _lock.lock();

            _references.emplace_back(&readable);

            _lock.unlock();

            // return readable;
        }
    }

    bool AveragingArrayProxy::isEmpty() {
        return _original->isEmpty();
    }

    NDArray* AveragingArrayProxy::writeable(int row, int key) {
        std::pair<int, int> k(row, key);

        // if writeable exists - just return it
        if (writeableExists(k)) {
            _lock.lock();

            auto r = _writeables[k];

            _lock.unlock();

            return r;
        } else {
            // if doesn't - let's create it
            auto orig = (*_original)({row,row+1, 0,0});

            // we don't want views here for obvious reasons
            auto writeable = orig.dup('c');            

            _lock.lock();

            _writeables[k] = writeable;
            _references.emplace_back(writeable);

            // storing linear reference, for future averaging step
            if (_writeablesLinear.count(row) == 0) {
                std::vector<NDArray*> vec;
                _writeablesLinear[row] = vec;
            }

            _writeablesLinear[row].emplace_back(writeable);
            _rows.emplace_back(row);

            _lock.unlock();

            return writeable;
        }
    }

    bool AveragingArrayProxy::collapseWrites() {
        if (_writeables.empty())
            return false;

        for (int r = 0; r < _rows.size(); r++) {
            auto row = _rows[r];
            auto list = _writeablesLinear.at(row);

            auto originalRow = (*_original)({row,row+1, 0,0});

            if (list.size() == 1) {
                originalRow.assign(list.at(0));
            } else {
                originalRow.assign(0.0);

                for (int e = 0; e < list.size(); e++)
                    originalRow += *(list.at(e));

                originalRow /= (int) list.size();
            }
        }

        return true;
    }
}