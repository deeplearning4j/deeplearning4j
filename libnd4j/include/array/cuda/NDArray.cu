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

/*
 * NDArray.cu has been split into multiple smaller files to reduce object file size
 * and avoid PC32 relocation overflow errors in .eh_frame sections during linking.
 *
 * The implementation is now distributed across:
 *   - NDArray_core.cu      : Core sync/buffer functions, basic operations
 *   - NDArray_triangular.cu: Triangular and identity matrix operations
 *   - NDArray_tile.cu      : Tile operations
 *   - NDArray_repeat.cu    : Repeat operations (BUILD_DOUBLE_TEMPLATE = 169 instantiations)
 *   - NDArray_print.cu     : Print buffer operations
 *
 * This file now only includes the shared header for the NDArray.hXX implementation.
 */

#ifndef NDARRAY_CPP
#define NDARRAY_CPP

#include <array/NDArray.h>
#include <array/NDArray.hXX>

#endif // NDARRAY_CPP
