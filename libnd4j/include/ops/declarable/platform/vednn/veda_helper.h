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

#ifndef DEV_VEDAHELPERS_H
#define DEV_VEDAHELPERS_H

#include <veda.h>
#include <helpers/logger.h>

#define VEDA(err) veda_check(err, __FILE__, __LINE__)

SD_INLINE void veda_check(VEDAresult err, const char* file, const int line) {
  if (err != VEDA_SUCCESS) {
    const char *name, *str;
    vedaGetErrorName(err, &name);
    vedaGetErrorString(err, &str);
    sd_printf("%s: %s @ %s:%i\n", name, str, file, line);
    exit(1);
  }
}

struct VEDA_HANDLE {
  VEDAcontext ctx;
  VEDAmodule mod;
  using FUNC_NAME_PTR = const char*;

  SD_MAP_IMPL<FUNC_NAME_PTR, VEDAfunction> functionsLookUp;

  static VEDA_HANDLE& getInstance() {
    static VEDA_HANDLE instance(VEDA_VEDNN_LIBRARY);
    return instance;
  }

  VEDAfunction getFunctionByConstPtrName(FUNC_NAME_PTR namePtr) {
    auto searchIter = functionsLookUp.find(namePtr);
    if (searchIter != functionsLookUp.end()) return searchIter->second;
    // insert to our lookUp
    VEDAfunction func;
    VEDA(vedaModuleGetFunction(&func, mod, namePtr));
    functionsLookUp.emplace(namePtr, func);
    return func;
  }

 private:
  VEDA_HANDLE(const char* library_name) {

    VEDA(vedaInit(0));
    VEDA(vedaCtxCreate(&ctx, 0, 0));
    if(const char* env_p = std::getenv("DEVICE_LIB_LOADPATH")){
        std::string path_lib = std::string(env_p) + "/" + library_name;
        VEDA(vedaModuleLoad(&mod, path_lib.c_str()));
    }else{
        VEDA(vedaModuleLoad(&mod, library_name));
    }
  }
  VEDA_HANDLE() = delete;
  VEDA_HANDLE(const VEDA_HANDLE&) = delete;
  VEDA_HANDLE(VEDA_HANDLE&&) = delete;
  VEDA_HANDLE& operator=(const VEDA_HANDLE&) = delete;
  VEDA_HANDLE& operator=(VEDA_HANDLE&&) = delete;

 protected:
  virtual ~VEDA_HANDLE() {
    //we are doing nothing ,for now
    //https://github.com/SX-Aurora/veda/issues/16
    //VEDA(vedaExit());
  }
};

#endif
