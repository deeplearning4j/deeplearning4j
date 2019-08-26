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

#ifndef LIBND4J_CONTEXTBUFFERS_H
#define LIBND4J_CONTEXTBUFFERS_H

#include <dll.h>
#include <pointercast.h>
#include <execution/ErrorReference.h>

namespace nd4j {
    class ND4J_EXPORT ContextBuffers {
    private:
        void* _reductionPointer = nullptr;
        void* _scalarPointer = nullptr;
        void* _allocationPointer = nullptr;
        void* _execStream = nullptr;
        void* _specialStream = nullptr;
        sd::ErrorReference _errorReference;
        bool _allocated = false;
        bool _initialized = false;

        int _deviceId = -1;

        void initialize();
    public:
        ContextBuffers();
        ContextBuffers(const ContextBuffers &other);
        ContextBuffers(void* rPointer, void* sPointer, void* aPointer, bool isOwner = false);
        ~ContextBuffers();

        ContextBuffers& operator=(const ContextBuffers& other);
        ContextBuffers& operator=(ContextBuffers&& other);

        void release();

        void* reductionBuffer();
        void* scalarBuffer();
        void* allocationBuffer();

        void* execStream();
        void* specialStream();

        void setReductionBuffer(void* pointer);
        void setScalarBuffer(void* pointer);
        void setAllocationBuffer(void* pointer);

        sd::ErrorReference* errorReference();

        void triggerOwnership(bool isOwner);

        int deviceId();

        bool isInitialized();
    };
}


#endif //DEV_TESTS_CONTEXTBUFFERS_H
