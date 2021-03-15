/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// Created by raver on 8/31/2018.
//

#ifndef DEV_TESTS_GRAPH_EXISTS_EXCEPTION_H
#define DEV_TESTS_GRAPH_EXISTS_EXCEPTION_H

#include <system/op_boilerplate.h>
#include <system/pointercast.h>
#include <stdexcept>
#include <exceptions/graph_exception.h>
#include <system/dll.h>

#if defined(_MSC_VER)

// we're ignoring warning about non-exportable parent class, since std::runtime_error is a part of Standard C++ Library
#pragma warning( disable : 4275 )

#endif

namespace sd {
    class ND4J_EXPORT graph_exists_exception: public graph_exception {
    public:
        explicit graph_exists_exception(Nd4jLong graphId);
    };
}

#endif //DEV_TESTS_UNKNOWN_GRAPH_EXCEPTION_H
