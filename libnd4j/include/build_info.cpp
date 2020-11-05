/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <config.h>
#include <build_info.h>

const char* buildInfo() {
	return ""
#if defined(__clang__)
        "Clang: " TOSTRING(__clang_version__)
#elif defined(_MSC_VER)
        "MSVC: " TOSTRING(_MSC_FULL_VER)
#else
        "GCC: " TOSTRING(__VERSION__)
#endif
#if defined(_MSC_VER) && defined(_MSVC_LANG)  
        "\nSTD version: " TOSTRING(_MSVC_LANG)
#elif defined(__cplusplus)
        "\nSTD version: " TOSTRING(__cplusplus)
#endif

#if defined(__CUDACC__)
        "\nCUDA: " TOSTRING(__CUDACC_VER_MAJOR__)
        "."  TOSTRING(__CUDACC_VER_MINOR__)
        "." TOSTRING(__CUDACC_VER_BUILD__)
#endif
#if defined(DEFAULT_ENGINE)
        "\nDEFAULT_ENGINE: " TOSTRING(DEFAULT_ENGINE)
#endif
#if defined(HAVE_FLATBUFFERS)
        "\nHAVE_FLATBUFFERS"
#endif
#if defined(HAVE_MKLDNN)
        "\nHAVE_MKLDNN"
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
