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

#ifndef DEV_TESTS_UTF8STRING_H
#define DEV_TESTS_UTF8STRING_H

#include <string>
#include <system/dll.h>

namespace sd {
    struct ND4J_EXPORT utf8string {
    private:
        bool _allocated = false;
    public:
        char *_buffer = nullptr;
        unsigned int _length = 0;

        utf8string();
        ~utf8string();

        utf8string(const char *string, int length);
        utf8string(const std::string &string);
        utf8string(const utf8string &other);
        utf8string& operator=(const utf8string &other);

    protected:
        void Swap(utf8string &other);
    };
}


#endif //DEV_TESTS_UTF8STRING_H
