/* ******************************************************************************
*
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
*  See the NOTICE file distributed with this work for additional
*  information regarding copyright ownership.
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See
* the License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

#ifndef SD_OPTIONAL_H_
#define SD_OPTIONAL_H_

#include "../system/common.h"
#include "../system/op_boilerplate.h"

namespace sd {

template<typename T>
class Optional {
 private:
  T value;
  bool hasValue;
 
 public:
  Optional() : hasValue(false) {}
  explicit Optional(T val) : value(val), hasValue(true) {}
  
  bool isPresent() const { return hasValue; }
  operator bool() const { return hasValue && value; }
  T get() const { 
    if (!hasValue) THROW_EXCEPTION("Accessing empty Optional"); 
    return value;
  }
  T getOrThrow(const char* msg) const {
    if (!hasValue) THROW_EXCEPTION(msg);
    return value;
  }
  
  operator T() const { return get(); }  // Implicit conversion to T
};

// Specialization for pointer types
template<typename T>
class Optional<T*> {
 private:
  T* value;
  bool hasValue;
 
 public:
  Optional() : value(nullptr), hasValue(false) {}
  explicit Optional(T* val) : value(val), hasValue(val != nullptr) {}
  
  bool isPresent() const { return hasValue && value != nullptr; }
  operator bool() const { return isPresent(); }
  T* get() const { 
    if (!isPresent()) THROW_EXCEPTION("Accessing empty Optional"); 
    return value;
  }
  T* getOrThrow(const char* msg) const {
    if (!isPresent()) THROW_EXCEPTION(msg);
    return value;
  }
  
  operator T*() const { return get(); }  // Implicit conversion to T*
  T* operator->() const { return get(); }
  T& operator*() const { return *get(); }
};

} // namespace sd

#endif // SD_OPTIONAL_H_