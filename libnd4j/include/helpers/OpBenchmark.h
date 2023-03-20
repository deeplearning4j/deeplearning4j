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
// Created by raver on 2/28/2019.
//

#ifndef DEV_TESTS_OPEXECUTIONER_H
#define DEV_TESTS_OPEXECUTIONER_H
#include <array/NDArray.h>
#include <helpers/StringUtils.h>
#include <helpers/ConstantTadHelper.h>
#include <helpers/PointersManager.h>
#include <helpers/ShapeUtils.h>
#include <legacy/NativeOpExecutioner.h>

namespace sd {
class SD_LIB_EXPORT OpBenchmark {
 protected:
  int _opNum = 0;
  std::string _testName;
  NDArray *_x = nullptr;
  NDArray *_y = nullptr;
  NDArray *_z = nullptr;
  std::vector<sd::LongType> _axis;

 public:
  OpBenchmark() = default;
  OpBenchmark(std::string name, NDArray *x, NDArray *y, NDArray *z);
  OpBenchmark(std::string name, NDArray *x, NDArray *z);
  OpBenchmark(std::string name, NDArray *x, NDArray *z, std::initializer_list<sd::LongType> axis);
  OpBenchmark(std::string name, NDArray *x, NDArray *z, std::vector<sd::LongType> axis);
  OpBenchmark(std::string name, NDArray *x, NDArray *y, NDArray *z, std::initializer_list<sd::LongType> axis);
  OpBenchmark(std::string name, NDArray *x, NDArray *y, NDArray *z, std::vector<sd::LongType> axis);

  void setOpNum(int opNum);
  void setTestName(std::string testName);
  void setX(NDArray *array);
  void setY(NDArray *array);
  void setZ(NDArray *array);
  void setAxis(std::vector<sd::LongType> axis);
  void setAxis(std::initializer_list<sd::LongType> axis);

  NDArray &x();
  int opNum();
  std::string testName();
  std::vector<sd::LongType> getAxis();

  virtual std::string extra();
  virtual std::string dataType();
  virtual std::string axis() = 0;
  virtual std::string orders() = 0;
  virtual std::string strides() = 0;
  virtual std::string shape();
  virtual std::string inplace() = 0;

  virtual void executeOnce() = 0;

  virtual OpBenchmark *clone() = 0;
};
}  // namespace sd

#endif  // DEV_TESTS_OPEXECUTIONER_H
