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
// Created by raver119 on 11/06/18.
//

#ifndef LIBND4J_RESULTWRAPPER_H
#define LIBND4J_RESULTWRAPPER_H
#include <system/common.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace graph {
class SD_LIB_EXPORT ResultWrapper {
 private:
  sd::LongType _size = 0L;
  sd::Pointer _pointer = nullptr;

 public:
  ResultWrapper(sd::LongType size, sd::Pointer ptr);
  ~ResultWrapper();

  sd::LongType size();

  sd::Pointer pointer();
};
}  // namespace graph
}  // namespace sd

#endif  // LIBND4J_RESULTWRAPPER_H
