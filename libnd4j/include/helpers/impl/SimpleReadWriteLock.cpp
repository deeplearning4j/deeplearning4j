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

#include <helpers/SimpleReadWriteLock.h>


namespace nd4j {
    void SimpleReadWriteLock::lockRead() {
        _mutex.lock();
        _read_locks++;
        while(_write_locks.load() > 0) {
            // just loop
        }
        _mutex.unlock();
    }

    void SimpleReadWriteLock::unlockRead() {
        _read_locks--;
    }

    // write lock
    void SimpleReadWriteLock::lockWrite() {
        _mutex.lock();
        _write_locks++;
        while (_read_locks.load() > 0) {
            // just loop
        }
        _mutex.unlock();
    }

    void SimpleReadWriteLock::unlockWrite() {
        _write_locks--;
    }

    SimpleReadWriteLock& SimpleReadWriteLock::operator= ( const SimpleReadWriteLock &other) {
        this->_write_locks.store(other._write_locks.load());
        this->_read_locks.store(other._read_locks.load());
    }
}