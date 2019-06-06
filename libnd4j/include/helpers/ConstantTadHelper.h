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
//  @author raver119@gmail.com
//


#ifndef DEV_TESTS_CONSTANTTADHELPER_H
#define DEV_TESTS_CONSTANTTADHELPER_H

#include <dll.h>
#include <pointercast.h>
#include <map>
#include <vector>
#include <mutex>
#include <array/ShapeDescriptor.h>
#include <array/TadDescriptor.h>
#include <array/TadPack.h>

namespace nd4j {
    class ND4J_EXPORT ConstantTadHelper {
    private:
        static ConstantTadHelper *_INSTANCE;

        std::mutex _mutex;
        std::vector<std::map<TadDescriptor, TadPack>> _cache;

        ConstantTadHelper();
    public:
        ~ConstantTadHelper() = default;

        static ConstantTadHelper* getInstance();

        TadPack& tadForDimensions(Nd4jLong *originalShape, const std::vector<int> &dimensions, const bool keepUnitiesInShape = false);
        TadPack& tadForDimensions(Nd4jLong *originalShape, int* dimensions, int dimLength, const bool keepUnitiesInShape = false);
        TadPack& tadForDimensions(Nd4jLong *originalShape, int dimensions, const bool keepUnitiesInShape = false);
        TadPack& tadForDimensions(ShapeDescriptor &descriptor, std::vector<int> &dimensions, const bool keepUnitiesInShape = false);
        TadPack& tadForDimensions(TadDescriptor &descriptor);
    };
}

#endif //DEV_TESTS_CONSTANTTADHELPER_H
