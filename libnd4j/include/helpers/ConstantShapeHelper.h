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

#ifndef DEV_TESTS_CONSTANTSHAPEHELPER_H
#define DEV_TESTS_CONSTANTSHAPEHELPER_H

#include <system/dll.h>
#include <system/pointercast.h>
#include <map>
#include <mutex>
#include <vector>
#include <array/ShapeDescriptor.h>
#include <array/ConstantDataBuffer.h>
#include <memory/Workspace.h>
#include <system/op_boilerplate.h>

namespace sd {

    class ND4J_EXPORT ConstantShapeHelper {
    private:
        static ConstantShapeHelper *_INSTANCE;

        std::mutex _mutex;
        std::vector<MAP_IMPL<ShapeDescriptor, ConstantDataBuffer>> _cache;


        ConstantShapeHelper();
    public:
        ~ConstantShapeHelper() = default;

        static ConstantShapeHelper* getInstance();


        ConstantDataBuffer bufferForShapeInfo(sd::DataType dataType, char order, const std::vector<Nd4jLong> &shape);
        ConstantDataBuffer bufferForShapeInfo(const ShapeDescriptor &descriptor);
        ConstantDataBuffer bufferForShapeInfo(const Nd4jLong *shapeInfo);
        ConstantDataBuffer bufferForShapeInfo(const sd::DataType dataType, const char order, const int rank, const Nd4jLong* shape);
        ConstantDataBuffer createShapeInfoWithUnitiesForBroadcast(const Nd4jLong* maxShapeInfo, const Nd4jLong* minShapeInfo, sd::memory::Workspace* workspace = nullptr, const std::vector<int> dimensions = {});


        Nd4jLong* emptyShapeInfo(const sd::DataType dataType);
        Nd4jLong* scalarShapeInfo(const sd::DataType dataType);
        Nd4jLong* vectorShapeInfo(const Nd4jLong length, const sd::DataType dataType);
        Nd4jLong* createShapeInfo(const ShapeDescriptor &descriptor);
        Nd4jLong* createShapeInfo(const sd::DataType dataType, const char order, const std::vector<Nd4jLong> &shape);
        Nd4jLong* createShapeInfo(const sd::DataType dataType, const char order, const int rank, const Nd4jLong* shape);
        Nd4jLong* createShapeInfo(const sd::DataType dataType, const Nd4jLong* shapeInfo);

        Nd4jLong* createFromExisting(Nd4jLong *shapeInfo, sd::memory::Workspace *workspace);
        Nd4jLong* createFromExisting(Nd4jLong *shapeInfo, bool destroyOriginal = true);

        bool checkBufferExistenceForShapeInfo(ShapeDescriptor &descriptor);


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

#endif //DEV_TESTS_CONSTANTSHAPEHELPER_H
