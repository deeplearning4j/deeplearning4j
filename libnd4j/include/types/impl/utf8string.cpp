/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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

namespace sd {
    utf8string::~utf8string() {
        if (_allocated)
            delete[] _buffer;
    }

    utf8string::utf8string() {
        _allocated = false;
        _length = 0;
        _buffer = nullptr;
    }

    utf8string::utf8string(const char *string, int length) {
        _length = length;
        _buffer = new char[_length];
        _allocated = true;
        std::memset(_buffer, 0, _length + 1);
        std::memcpy(_buffer, string, _length);
    }

    utf8string::utf8string(const std::string &str) {
        _length = str.length();
        _buffer = new char[_length + 1];
        _allocated = true;
        std::memset(_buffer, 0, _length + 1);
        std::memcpy(_buffer, str.data(), _length);
        _buffer[_length] = 0;
    }

    utf8string::utf8string(const utf8string &other) {
        _length = other._length;
        _buffer = new char[_length+1];
        _allocated = true;
        std::memset(_buffer, 0, _length + 1);
        std::memcpy(_buffer, other._buffer, _length);
        _buffer[_length] = 0;
    }

    void utf8string::Swap(utf8string &other) {
        std::swap(_length, other._length);
        std::swap(_buffer, other._buffer);// = new char[_length+1];
        std::swap(_allocated, other._allocated); // = true;
    }

    utf8string& utf8string::operator=(const utf8string &other) {
        if (this != &other) {
            utf8string temp(other);
            Swap(temp);
        }
        return *this;
    }
}
