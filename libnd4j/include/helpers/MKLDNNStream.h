/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// Created by saudet on 8/30/2018.
//

#ifndef LIBND4J_ONEDNNSTREAM_H
#define LIBND4J_ONEDNNSTREAM_H


#if !defined(__STANDALONE_BUILD__)
#include "config.h"
#endif


#if HAVE_ONEDNN
#include <vector>
#include <string>
#include <array/NDArray.h>

namespace sd {
class ONEDNNStream {
 protected:
  std::string _opName;

  std::vector<NDArray *> _inputs;
  std::vector<NDArray *> _outputs;
  std::vector<float> _floatArguments;
  std::vector<int> _intArguments;

 public:
  template <typename X, typename Y>
  static bool isSupported() {
    // FIXME: strict float support doesn't work anymore
    return typeid(X) == typeid(float) && typeid(Y) == typeid(float);
  }

  static bool isSupported(const std::vector<NDArray *> &arrays) {
    // FIXME: strict float support doesn't work anymore
    for (auto v : arrays) {
      if (v != nullptr && v->dataType() != sd::DataType::FLOAT32) {
        return false;
      }
    }
    return true;
  }

  explicit ONEDNNStream(const std::string &opName) : _opName(opName) {}

  bool checkAndReset(const std::vector<NDArray *> &inputs, const std::vector<NDArray *> &outputs,
                     const std::vector<float> &floatArguments, const std::vector<int> &intArguments) {
    if (inputs != _inputs || outputs != _outputs || floatArguments != _floatArguments ||
        intArguments != _intArguments) {
      _inputs = inputs;
      _outputs = outputs;
      _floatArguments = floatArguments;
      _intArguments = intArguments;
      return true;
    }
    return false;
  }
};
}  // namespace sd
#endif

#endif  // LIBND4J_ONEDNNSTREAM_H
