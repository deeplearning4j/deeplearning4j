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

#include <types/utf8string.h>
#include <cstring>

namespace nd4j {
    utf8string::~utf8string() {
        if (_allocated)
            delete[] _buffer;
    }

    utf8string::utf8string() {
        _allocated = false;
        _length = 0;
        _buffer = nullptr;
    }

    utf8string::utf8string(std::string *str) {
        _length = str->length();
        _buffer = new char[_length + 1];
        _allocated = true;
        std::memcpy(_buffer, str->data(), _length);
        _buffer[_length] = 0;
    }

    utf8string::utf8string(const utf8string &other) {
        _length = other._length;
        _buffer = new char[_length+1];
        _allocated = true;
        std::memcpy(_buffer, other._buffer, _length);
        _buffer[_length] = 0;
    }

    utf8string& utf8string::operator=(const utf8string &other) {
//        if (_allocated && _length > 0)
//            delete[] _buffer;

        _length = other._length;
        _buffer = new char[_length+1];
        _allocated = true;
        std::memcpy(_buffer, other._buffer, _length);
        _buffer[_length] = 0;
    }
}
