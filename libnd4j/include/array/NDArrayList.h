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
// This class describes collection of NDArrays
//
// @author raver119!gmail.com
//

#ifndef NDARRAY_LIST_H
#define NDARRAY_LIST_H

#include <string>
#include <atomic>
#include <map>
#include <NDArray.h>
#include <memory/Workspace.h>

namespace nd4j {
    template <typename T>
    class NDArrayList {
    private:
        // workspace where chunks belong to
        nd4j::memory::Workspace* _workspace = nullptr;

        // numeric and symbolic ids of this list
        std::pair<int, int> _id;
        std::string _name;

        // stored chunks
        std::map<int, nd4j::NDArray<T>*> _chunks;

        // just a counter, for stored elements
        std::atomic<int> _elements;
        std::atomic<int> _counter;

        // reference shape
        std::vector<Nd4jLong> _shape;

        // unstack axis
        int _axis = 0;

        //
        bool _expandable = false;

        // maximum number of elements
        int _height = 0;
    public:
        NDArrayList(int height, bool expandable = false);
        ~NDArrayList();

        NDArray<T>* read(int idx);
        NDArray<T>* readRaw(int idx);
        Nd4jStatus write(int idx, NDArray<T>* array);

        NDArray<T>* pick(std::initializer_list<int> indices);
        NDArray<T>* pick(std::vector<int>& indices);
        bool isWritten(int index);

        NDArray<T>* stack();
        void unstack(NDArray<T>* array, int axis);

        std::pair<int,int>& id();
        std::string& name();
        nd4j::memory::Workspace* workspace();

        NDArrayList<T>* clone();

        bool equals(NDArrayList<T>& other);

        int elements();
        int height();

        int counter();
    };
}

#endif