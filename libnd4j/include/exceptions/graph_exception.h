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
// Created by raver on 9/1/2018.
//

#ifndef LIBND4J_GRAPH_EXCEPTION_H
#define LIBND4J_GRAPH_EXCEPTION_H

#include <system/common.h>

#include <stdexcept>
#include <string>

namespace sd {
class SD_LIB_EXPORT graph_exception : public std::runtime_error {
 protected:
  LongType _graphId;
  std::string _message;
  std::string _description;

 public:
  graph_exception(std::string message, LongType graphId);
  graph_exception(std::string message, std::string description, LongType graphId);
  graph_exception(std::string message, const char *description, LongType graphId);
  ~graph_exception() = default;

  LongType graphId();

  const char *message();
  const char *description();
};
}  // namespace sd

#endif  // DEV_TESTS_GRAPH_EXCEPTION_H
