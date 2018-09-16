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
// Created by raver119 on 2018-09-16.
//

#ifndef DEV_TESTS_NDARRAYFACTORY_H
#define DEV_TESTS_NDARRAYFACTORY_H

#include <vector>
#include <initializer_list>
#include <NDArray.h>
#include <memory/Workspace.h>


namespace nd4j {
    class NDArrayFactory {
    public:
        template <typename T>
        static NDArray* empty(nd4j::memory::Workspace* workspace = nullptr);

        template <typename T>
        static NDArray* valueOf(const std::initializer_list<Nd4jLong>& shape, const T value, const char order = 'c');

        template <typename T>
        static NDArray* valueOf(const std::vector<Nd4jLong>& shape, const T value, const char order = 'c');

        static NDArray* valueOf(const std::vector<Nd4jLong>& shape, const NDArray& value, const char order = 'c');

        template <typename T>
        static NDArray* linspace(const T from, const T to, const Nd4jLong numElements);

        template <typename T>
        static NDArray* scalar(const T value, nd4j::memory::Workspace* workspace = nullptr);

        static NDArray* scalar(nd4j::DataType dType, const double value, nd4j::memory::Workspace* workspace = nullptr);
        static NDArray* scalar(nd4j::DataType dType, const Nd4jLong value, nd4j::memory::Workspace* workspace = nullptr);

        template <typename T>
        static NDArray v(const T value, nd4j::memory::Workspace* workspace = nullptr);

        template <typename T>
        static NDArray* vector(Nd4jLong length, const T startingValue = (T) 0, nd4j::memory::Workspace *workspace = nullptr);

        template <typename T>
        static NDArray* create(std::initializer_list<Nd4jLong> s, nd4j::memory::Workspace* workspace);

        template <typename T>
        static NDArray* create(const char order, const std::vector<Nd4jLong> &shape, nd4j::memory::Workspace* workspace);

        template <typename T>
        static NDArray* p(const char order, const std::vector<Nd4jLong> &shape, const std::vector<T> &data, nd4j::memory::Workspace* workspace);

        template <typename T>
        static NDArray v(const char order, const std::vector<Nd4jLong> &shape, const std::vector<T> &data, nd4j::memory::Workspace* workspace);

#ifndef __JAVACPP_HACK__
        // this method only available out of javacpp
        /**
         * This constructor creates vector of T
         *
         * @param values
         */
        template <typename T>
        static NDArray* create(std::initializer_list<T> values, nd4j::memory::Workspace* workspace = nullptr);

        template <typename T>
        static NDArray* create(std::vector<T> &values, nd4j::memory::Workspace* workspace = nullptr);
#endif
    };
}

#endif //DEV_TESTS_NDARRAYFACTORY_H
