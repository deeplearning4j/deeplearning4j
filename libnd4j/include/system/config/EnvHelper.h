/* ******************************************************************************
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

#ifndef LIBND4J_ENV_HELPER_H
#define LIBND4J_ENV_HELPER_H

#include <cstdlib>
#include <string>
#include <stdexcept>

namespace sd {
namespace config {

/**
 * Utility functions for parsing environment variables in config subsystems.
 * Eliminates the repetitive try/catch + ifdef __cpp_exceptions boilerplate.
 */

inline bool readBoolEnv(const char* name, bool defaultValue) {
  const char* v = std::getenv(name);
  if (v == nullptr) return defaultValue;
  std::string s(v);
  if (s == "1" || s == "true" || s == "TRUE" || s == "ON" || s == "yes") return true;
  if (s == "0" || s == "false" || s == "FALSE" || s == "OFF" || s == "no") return false;
  return defaultValue;
}

inline int readBoolEnvTriState(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr) return -1;  // not set
  std::string s(v);
  if (s.empty()) return -1;  // not set — preserve C++ default
  if (s == "1" || s == "true" || s == "TRUE" || s == "ON" || s == "yes") return 1;
  return 0;
}

inline int readIntEnv(const char* name, int defaultValue) {
  const char* v = std::getenv(name);
  if (v == nullptr) return defaultValue;
#ifdef __cpp_exceptions
  try {
    return std::stoi(std::string(v));
  } catch (...) {
    return defaultValue;
  }
#else
  return std::stoi(std::string(v));
#endif
}

inline int64_t readInt64Env(const char* name, int64_t defaultValue) {
  const char* v = std::getenv(name);
  if (v == nullptr) return defaultValue;
#ifdef __cpp_exceptions
  try {
    return std::stoll(std::string(v));
  } catch (...) {
    return defaultValue;
  }
#else
  return std::stoll(std::string(v));
#endif
}

inline size_t readSizeTEnv(const char* name, size_t defaultValue) {
  const char* v = std::getenv(name);
  if (v == nullptr) return defaultValue;
#ifdef __cpp_exceptions
  try {
    return std::stoul(std::string(v));
  } catch (...) {
    return defaultValue;
  }
#else
  return std::stoul(std::string(v));
#endif
}

inline float readFloatEnv(const char* name, float defaultValue) {
  const char* v = std::getenv(name);
  if (v == nullptr) return defaultValue;
#ifdef __cpp_exceptions
  try {
    return std::stof(std::string(v));
  } catch (...) {
    return defaultValue;
  }
#else
  return std::stof(std::string(v));
#endif
}

inline std::string readStringEnv(const char* name, const std::string& defaultValue = "") {
  const char* v = std::getenv(name);
  if (v == nullptr) return defaultValue;
  return std::string(v);
}

}  // namespace config
}  // namespace sd

#endif  // LIBND4J_ENV_HELPER_H
