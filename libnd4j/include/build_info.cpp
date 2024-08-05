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
#include <build_info.h>
#include <config.h>

#include <string>

#include "helpers/logger.h"

#if defined(SD_GCC_FUNCTRACE)

bool isFuncTrace() {
  return true;
}

#else

bool isFuncTrace() {
  return false;
}

#endif

const char *buildInfo() {
  std::string ret = "Build Info: ";
#if defined(SD_GCC_FUNCTRACE)
  ret += "\nFunctrace: ";
  ret += isFuncTrace() ? "ON\n" : "OFF";

#endif

#if defined(__clang__)
  ret += "Clang: " STRINGIZE(__clang_version__);
#elif defined(_MSC_VER)
  ret += "MSVC: " STRINGIZE(_MSC_FULL_VER);
#elif defined(__NEC__)
  ret += "Nec CC: " STRINGIZE(__VERSION__);
#else
  ret += "GCC: " STRINGIZE(__VERSION__);
#endif
#if defined(_MSC_VER) && defined(_MSVC_LANG)
  ret +=      "\nSTD version: " STRINGIZE(_MSVC_LANG);
#elif defined(__cplusplus)
  ret += "\nSTD version: " STRINGIZE(__cplusplus);
#endif

#if defined(__CUDACC__)
  ret +=  "\nCUDA: " STRINGIZE(__CUDACC_VER_MAJOR__) "." STRINGIZE(__CUDACC_VER_MINOR__) "." STRINGIZE(;
                     __CUDACC_VER_BUILD__)
#endif
#if defined(DEFAULT_ENGINE)
  ret +=        "\nDEFAULT_ENGINE: " STRINGIZE(DEFAULT_ENGINE);
#endif
#if defined(HAVE_FLATBUFFERS)
  ret +=   "\nHAVE_FLATBUFFERS";
#endif
#if defined(HAVE_ONEDNN)
  ret +=   "\nHAVE_ONEDNN";
#endif
#if defined(HAVE_VEDNN)
  ret +=     "\nHAVE_VEDNN";
#endif
#if defined(__EXTERNAL_BLAS__)
  ret +=      "\nHAVE_EXTERNAL_BLAS";
#endif
#if defined(HAVE_OPENBLAS)
  ret += "\nHAVE_OPENBLAS";
#endif
#if defined(HAVE_CUDNN)
  ret +=  "\nHAVE_CUDNN";
#endif
#if defined(HAVE_ARMCOMPUTE)
  ret +=  "\nHAVE_ARMCOMPUTE";
#endif

#if defined(SD_CUDA)

  ret +=  "\nCUDA: " STRINGIZE(__CUDACC_VER_MAJOR__) "." STRINGIZE(__CUDACC_VER_MINOR__) "." STRINGIZE(
      __CUDACC_VER_BUILD__);


#endif
#if defined(CUDA_ARCHITECTURES)
  ret += "\nCUDA_ARCHITECTURES: " STRINGIZE(CUDA_ARCHITECTURES);
#endif




  std::string *ret2 =  new std::string(ret);
  //risk of build information not being printed during debug settings
  if(isFuncTrace())
    sd_printf("%s", ret2->c_str());
  return ret2->c_str();
}


