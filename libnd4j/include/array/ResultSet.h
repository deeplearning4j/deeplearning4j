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
// This class is suited for execution results representation. 
// 
// PLESE NOTE: It will delete all stored NDArrays upon destructor call
//
// Created by raver119 on 07.09.17.
//

#ifndef LIBND4J_RESULTSET_H
#define LIBND4J_RESULTSET_H

#include <vector>
#include <graph/generated/result_generated.h>
#include <pointercast.h>

namespace  nd4j {

    class NDArray; // forward declaration of template class NDArray
    
    class ResultSet {
    private:
        std::vector<nd4j::NDArray *> _content;
        Nd4jStatus _status = ND4J_STATUS_OK;
        bool _removable = true;

    public:
        // default constructor
        ResultSet(const nd4j::graph::FlatResult* result = nullptr);
        ~ResultSet();

        int size();
        nd4j::NDArray* at(unsigned long idx);
        void push_back(nd4j::NDArray* array);

        Nd4jStatus status();
        void setStatus(Nd4jStatus status);
        void purge();
        void setNonRemovable();
    };
}

#endif //LIBND4J_RESULTSET_H
