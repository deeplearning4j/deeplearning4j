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
    class ND4J_EXPORT NDArrayFactory {
    private:
        template <typename T>
        static void memcpyFromVector(void *ptr, const std::vector<T> &vector);
    public:
        template <typename T>
        static NDArray* empty(nd4j::memory::Workspace* workspace = nullptr);

        template <typename T>
        static NDArray* valueOf(const std::initializer_list<Nd4jLong>& shape, const T value, const char order = 'c',  nd4j::memory::Workspace* workspace = nullptr);

        template <typename T>
        static NDArray* valueOf(const std::vector<Nd4jLong>& shape, const T value, const char order = 'c',  nd4j::memory::Workspace* workspace = nullptr);

        static NDArray* valueOf(const std::vector<Nd4jLong>& shape, const NDArray& value, const char order = 'c',  nd4j::memory::Workspace* workspace = nullptr);

        template <typename T>
        static NDArray* linspace(const T from, const T to, const Nd4jLong numElements);

        template <typename T>
        static NDArray* scalar_(const T value, nd4j::memory::Workspace* workspace = nullptr);
        template <typename T>
        static NDArray* create_(const T value, nd4j::memory::Workspace* workspace = nullptr);

        template <typename T>
        static NDArray scalar(const T value, nd4j::memory::Workspace* workspace = nullptr);

        template <typename T>
        static NDArray create(const T value, nd4j::memory::Workspace* workspace = nullptr);
        static NDArray create(nd4j::DataType dtype, nd4j::memory::Workspace* workspace = nullptr);


        template <typename T>
        static NDArray* vector(Nd4jLong length, const T startingValue = (T) 0, nd4j::memory::Workspace *workspace = nullptr);

        template <typename T>
        static NDArray* create_(const char order, const std::vector<Nd4jLong> &shape, nd4j::memory::Workspace* workspace = nullptr);

        static NDArray* create_( const char order, const std::vector<Nd4jLong> &shape, nd4j::DataType dataType, nd4j::memory::Workspace* workspace = nullptr);

        template <typename T>
        static NDArray* create_(const char order, const std::vector<Nd4jLong> &shape, const std::vector<T> &data, nd4j::memory::Workspace* workspace = nullptr);

        template <typename T>
        static NDArray create(const char order, const std::vector<Nd4jLong> &shape, const std::vector<T> &data, nd4j::memory::Workspace* workspace = nullptr);

        template <typename T>
        static NDArray create(const char order, const std::vector<Nd4jLong> &shape, nd4j::memory::Workspace* workspace = nullptr);
        static NDArray create(const char order, const std::vector<Nd4jLong> &shape, nd4j::DataType dtype, nd4j::memory::Workspace* workspace = nullptr);

        template <typename T>
        static NDArray create(const std::vector<T> &values, nd4j::memory::Workspace* workspace = nullptr);

#ifndef __JAVACPP_HACK__
        // this method only available out of javacpp
        /**
         * This constructor creates vector of T
         *
         * @param values
         */

        template <typename T>
        static NDArray create(const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::memory::Workspace* workspace = nullptr);

        template <typename T>
        static NDArray create(T* buffer, const char order, const std::initializer_list<Nd4jLong>& shape, nd4j::memory::Workspace* workspace = nullptr);

        template <typename T>
        static NDArray create(const char order, const std::vector<Nd4jLong> &shape, const std::initializer_list<T>& data, nd4j::memory::Workspace* workspace = nullptr);
#endif
    };
}

#endif //DEV_TESTS_NDARRAYFACTORY_H
