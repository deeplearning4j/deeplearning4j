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

#include <system/dll.h>
#include <system/op_boilerplate.h>
#include <system/pointercast.h>
#include <map>
#include <vector>
#include <mutex>
#include <array/ShapeDescriptor.h>
#include <array/TadDescriptor.h>
#include <array/TadPack.h>

namespace sd {
    class ND4J_EXPORT ConstantTadHelper {
    private:
        std::mutex _mutex;
        std::vector<MAP_IMPL<TadDescriptor, TadPack>> _cache;

        ConstantTadHelper();
    public:
        ~ConstantTadHelper() = default;

        static ConstantTadHelper & getInstance();

        /**
         * These methods calculate Tensor-Along-Dimension(s) shape and offsets
         *
         * @param originalShape
         * @param dimensions
         * @param keepUnitiesInShape
         * @return
         */
        TadPack tadForDimensions(const Nd4jLong *originalShape, const std::vector<int> &dimensions, const bool keepUnitiesInShape = false);
        TadPack tadForDimensions(const Nd4jLong *originalShape, int* dimensions, int dimLength, const bool keepUnitiesInShape = false);
        TadPack tadForDimensions(const Nd4jLong *originalShape, int dimensions, const bool keepUnitiesInShape = false);
        TadPack tadForDimensions(ShapeDescriptor &descriptor, std::vector<int> &dimensions, const bool keepUnitiesInShape = false);
        TadPack tadForDimensions(TadDescriptor &descriptor);

        /**
         * This method returns number of cached TAD shapes/offsets on specific device
         * @return
         */
        FORCEINLINE int cachedEntriesForDevice(int deviceId) {
            if (deviceId > _cache.size())
                throw std::runtime_error("deviceId > number of actual devices");

            return _cache[deviceId].size();
        }

        /**
         * This method returns total number of cached TAD shapes/offsets on all devices
         * @return
         */
        FORCEINLINE int totalCachedEntries() {
            int total = 0;

            for (int e = 0; e < _cache.size(); e++)
                total += _cache[e].size();

            return total;
        }
    };
}

#endif //DEV_TESTS_CONSTANTTADHELPER_H
