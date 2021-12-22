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

const char *buildInfo() {
  return ""
#if defined(__clang__)
         "Clang: " STRINGIZE(__clang_version__)
#elif defined(_MSC_VER)
         "MSVC: " STRINGIZE(_MSC_FULL_VER)
#elif defined(__NEC__)
         "Nec CC: " STRINGIZE(__VERSION__)
#else
         "GCC: " STRINGIZE(__VERSION__)
#endif
#if defined(_MSC_VER) && defined(_MSVC_LANG)
             "\nSTD version: " STRINGIZE(_MSVC_LANG)
#elif defined(__cplusplus)
             "\nSTD version: " STRINGIZE(__cplusplus)
#endif

#if defined(__CUDACC__)
                 "\nCUDA: " STRINGIZE(__CUDACC_VER_MAJOR__) "." STRINGIZE(__CUDACC_VER_MINOR__) "." STRINGIZE(
                     __CUDACC_VER_BUILD__)
#endif
#if defined(DEFAULT_ENGINE)
                     "\nDEFAULT_ENGINE: " STRINGIZE(DEFAULT_ENGINE)
#endif
#if defined(HAVE_FLATBUFFERS)
                         "\nHAVE_FLATBUFFERS"
#endif
#if defined(HAVE_ONEDNN)
                         "\nHAVE_ONEDNN"
#endif
#if defined(HAVE_VEONEDNN)
                         "\nHAVE_VEONEDNN"
#endif
#if defined(HAVE_VEDNN)
                         "\nHAVE_VEDNN"
#endif
#if defined(__EXTERNAL_BLAS__)
                         "\nHAVE_EXTERNAL_BLAS"
#endif
#if defined(HAVE_OPENBLAS)
                         "\nHAVE_OPENBLAS"
#endif
#if defined(HAVE_CUDNN)
                         "\nHAVE_CUDNN"
#endif
#if defined(HAVE_ARMCOMPUTE)
                         "\nHAVE_ARMCOMPUTE"
#endif
      ;
}
