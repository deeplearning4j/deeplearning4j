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


#ifndef DEV_TESTS_AVERAGINGARRAYPROXY_H
#define DEV_TESTS_AVERAGINGARRAYPROXY_H

#include "NDArray.h"
#include <utility>
#include <map>
#include <vector>
#include <mutex>

namespace nd4j {
    class ND4J_EXPORT AveragingArrayProxy {
    protected:
        NDArray *_original;

        std::map<std::pair<int,int>, NDArray*> _writeables;
        std::map<int, std::vector<NDArray*>> _writeablesLinear;
        std::vector<int> _rows;

        std::vector<NDArray*> _references;

        std::mutex _lock;
    public:
        explicit AveragingArrayProxy(NDArray *original);
        ~AveragingArrayProxy();

        NDArray* readable(int row, int key);
        NDArray* writeable(int row, int key);

        bool isEmpty();

        bool writeableExists(std::pair<int, int> &key);
        bool writeableExists(int row, int key);

        bool collapseWrites();
    };
}

#endif //DEV_TESTS_AVERAGINGARRAYPROXY_H
