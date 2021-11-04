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
// Created by agibsonccc on 1/15/17.
//

#ifndef LIBND4J_TESTINCLUDE_H
#define LIBND4J_TESTINCLUDE_H
#include <system/op_boilerplate.h>

#include <string>

#include "testlayers.h"

// https://stackoverflow.com/questions/228005/alternative-to-itoa-for-converting-integer-to-string-c
SD_INLINE std::string int_array_to_string(sd::LongType int_array[], sd::LongType size_of_array) {
  std::string returnstring = "[";
  for (int temp = 0; temp < size_of_array; temp++) {
    returnstring += std::to_string(int_array[temp]);
    if (temp < size_of_array - 1) returnstring += ",";
  }
  returnstring += "]";
  return returnstring;
}

SD_INLINE ::testing::AssertionResult arrsEquals(sd::LongType n, sd::LongType *assertion, sd::LongType *other) {
  for (int i = 0; i < n; i++) {
    if (assertion[i] != other[i]) {
      std::string message = std::string("Failure at index  ") + std::to_string(i) + std::string(" assertion: ") +
                            int_array_to_string(assertion, n) + std::string(" and test array ") +
                            int_array_to_string(other, n) + std::string("  is not equal");
      return ::testing::AssertionFailure() << message;
    }
  }
  return ::testing::AssertionSuccess();
}

#endif  // LIBND4J_TESTINCLUDE_H
