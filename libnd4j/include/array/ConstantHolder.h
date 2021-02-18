/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author raver119@gmail.com
//

#ifndef LIBND4J_CONSTANTHOLDER_H
#define LIBND4J_CONSTANTHOLDER_H

#include <map>
#include <array/ConstantDescriptor.h>
#include <array/ConstantDataBuffer.h>
#include <mutex>

namespace sd {
    class ConstantHolder {
    private:
        int _deviceId = 0;
        std::mutex _mutex;

        std::map<sd::DataType, ConstantDataBuffer> _buffers;
    public:
        ConstantHolder(const ConstantHolder& other);
        ConstantHolder() = default;
        ~ConstantHolder() = default;

        ConstantHolder& operator=(const ConstantHolder& other) = default;
        ConstantHolder& operator=(ConstantHolder&& other) = default;

        bool hasBuffer(sd::DataType dataType);

        template <typename T>
        bool hasBuffer();

        void addBuffer(ConstantDataBuffer &pointer, sd::DataType dataType);

        template <typename T>
        void addBuffer(ConstantDataBuffer &pointer);

        ConstantDataBuffer* getConstantDataBuffer(sd::DataType dataType);

        template <typename T>
        ConstantDataBuffer* getConstantDataBuffer();

        std::mutex* mutex();
    };
}


#endif //DEV_TESTS_CONSTANTHOLDER_H
