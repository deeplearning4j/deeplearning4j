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
#include <helpers/helper_hash.h>
#include <helpers/logger.h>

namespace sd {
namespace ops {

HashHelper& HashHelper::getInstance() {
  static HashHelper instance;
  return instance;
}

LongType HashHelper::getLongHash(std::string& str) {
  _locker.lock();
  if (!_isInit) {
    sd_verbose("Building HashUtil table\n", "");

    unsigned long long h = 0x544B2FBACAAF1684L;
    for (int i = 0; i < 256; i++) {
      for (int j = 0; j < 31; j++) {
        h = (((unsigned long long)h) >> 7) ^ h;
        h = (h << 11) ^ h;
        h = (((unsigned long long)h) >> 10) ^ h;
      }
      _byteTable[i] = h;
    }

    _isInit = true;
  }

  _locker.unlock();

  //note: DO NOT change this type.
  //when something like thread sanitizer + cuda is used
  //the offsets can get absurdly big.
  //you get errors like: left shift of 11 places cannot be represented in type
  unsigned long long h = HSTART;
  unsigned long long hmult = HMULT;

  LongType len = str.size();
  for (int i = 0; i < len; i++) {
    char ch = str.at(i);
    auto uch = (unsigned char)ch;
    h = (h * hmult) ^ _byteTable[ch & 0xff];
    h = (h * hmult) ^ _byteTable[(uch >> 8) & 0xff];
  }

  return h;
}
}  // namespace ops
}  // namespace sd
