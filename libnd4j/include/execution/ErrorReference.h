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

#ifndef DEV_TESTS_ERRORREFERENCE_H
#define DEV_TESTS_ERRORREFERENCE_H

#include <string>
#include <system/dll.h>

namespace sd {
    class ND4J_EXPORT ErrorReference {
    private:
        int _errorCode = 0;
        std::string _errorMessage;
    public:
        ErrorReference() = default;
        ~ErrorReference() = default;

        int errorCode();
        const char* errorMessage();

        void setErrorCode(int errorCode);
        void setErrorMessage(std::string message);
        void setErrorMessage(const char* message);
    };
}


#endif //DEV_TESTS_ERRORREFERENCE_H
