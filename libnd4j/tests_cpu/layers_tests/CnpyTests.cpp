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
// Created by agibsonccc on 3/30/17.
//
#include <legacy/NativeOps.h>

#include <string>

#include "testinclude.h"

class FileTest : public NDArrayTests {};

class LoadFromStringTest : public NDArrayTests {};

class HeaderTest : public NDArrayTests {};

TEST_F(HeaderTest, test_dataTypes_1) {
  std::string header("0NUMPY6789{'descr': '>f4");

  ASSERT_EQ(sd::DataType::FLOAT32, dataTypeFromNpyHeader(const_cast<char *>(header.data())));
}

TEST_F(HeaderTest, test_dataTypes_2) {
  std::string header("0NUMPY6789{'descr': '>f8");

  ASSERT_EQ(sd::DataType::DOUBLE, dataTypeFromNpyHeader(const_cast<char *>(header.data())));
}

TEST_F(HeaderTest, test_dataTypes_3) {
  std::string header("0NUMPY6789{'descr': '<i4");

  ASSERT_EQ(sd::DataType::INT32, dataTypeFromNpyHeader(const_cast<char *>(header.data())));
}

TEST_F(HeaderTest, test_dataTypes_4) {
  std::string header("0NUMPY6789{'descr': '>u2");

  ASSERT_EQ(sd::DataType::UINT16, dataTypeFromNpyHeader(const_cast<char *>(header.data())));
}

