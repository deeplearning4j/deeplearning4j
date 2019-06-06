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

#include <helpers/helper_hash.h>
#include <helpers/logger.h>

namespace nd4j {
    namespace ops {

        HashHelper* HashHelper::getInstance() {
            if (_INSTANCE == 0)
                _INSTANCE = new HashHelper();

            return _INSTANCE;
        }

        Nd4jLong HashHelper::getLongHash(std::string& str) {
            _locker.lock();
            if (!_isInit) {
                nd4j_verbose("Building HashUtil table\n","");

                Nd4jLong h = 0x544B2FBACAAF1684L;
                for (int i = 0; i < 256; i++) {
                    for (int j = 0; j < 31; j++) {
                        h = (((unsigned long long) h) >> 7) ^ h;
                        h = (h << 11) ^ h;
                        h = (((unsigned long long) h) >> 10) ^ h;
                    }
                    _byteTable[i] = h;
                }


                _isInit = true;
            }

            _locker.unlock();

            Nd4jLong h = HSTART;
            Nd4jLong hmult = HMULT;
            Nd4jLong len = str.size();
            for (int i = 0; i < len; i++) {
                char ch = str.at(i);
                auto uch = (unsigned char) ch;
                h = (h * hmult) ^ _byteTable[ch & 0xff];
                h = (h * hmult) ^ _byteTable[(uch >> 8) & 0xff];
            }

            return h;
        }

        nd4j::ops::HashHelper* nd4j::ops::HashHelper::_INSTANCE = 0;
    }
}

