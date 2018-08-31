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
// Created by raver on 8/29/2018.
//

#ifndef LIBND4J_READWRITELOCK_H
#define LIBND4J_READWRITELOCK_H

#include <atomic>
#include <mutex>

/**
 * This class provides PRIMITIVE read-write lock, and should NOT be used outside of GraphServer due to its inefficiency.
 * However, since GraphServer isn't supposed to have Reads/Writes ration even close to 1.0, it'll work just fine.
 *
 * Basic idea: write lock won't be obtained before all read requests served
 */
namespace nd4j {
    class SimpleReadWriteLock {
    private:
        std::atomic<unsigned long long int> _read_locks;
        std::atomic<unsigned long long int> _write_locks;
        std::mutex _mutex;

    public:
        SimpleReadWriteLock();
        ~SimpleReadWriteLock() = default;

        // read lock
        void lockRead();
        void unlockRead();

        // write lock
        void lockWrite();
        void unlockWrite();

        SimpleReadWriteLock& operator= ( const SimpleReadWriteLock &other);
    };
}


#endif //DEV_TESTS_READWRITELOCK_H
