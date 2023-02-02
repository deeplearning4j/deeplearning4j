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
// This class is suited for execution results representation.
//
// PLESE NOTE: It will delete all stored NDArrays upon destructor call
//
// @author raver119@gmail.com
//

#ifndef LIBND4J_RESULTSET_H
#define LIBND4J_RESULTSET_H
#include <graph/scheme/result_generated.h>
#include <system/common.h>

#include <vector>

namespace sd {

class NDArray;  // forward declaration of template class NDArray

class SD_LIB_EXPORT ResultSet {
 private:
  std::vector<sd::NDArray *> _content;
  sd::Status _status = sd::Status::OK;
  bool _removable = true;

  void delContent();

 public:
  explicit ResultSet();

#ifndef __JAVACPP_HACK__
  ResultSet(const sd::graph::FlatResult *result);
#endif

  ResultSet(const ResultSet &other) noexcept;

  ResultSet &operator=(const ResultSet &other) noexcept;

  // move constructor
  ResultSet(ResultSet &&other) noexcept;

  // move assignment operator
  ResultSet &operator=(ResultSet &&other) noexcept;

  ~ResultSet();

  int size();
  sd::NDArray *at(const unsigned long idx) const;
  sd::NDArray *operator[](const unsigned long idx) const;
  void push_back(sd::NDArray *array);

  sd::Status status();
  void setStatus(sd::Status status);
  void purge();
  void setNonRemovable();
};
}  // namespace sd

#endif  // LIBND4J_RESULTSET_H
