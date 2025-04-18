//
// Created by agibsonccc on 4/11/25.
//

/* ******************************************************************************
*
* Copyright (c) 2024 Konduit K.K.
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

//
// @author Adam Gibson
//

#ifndef LIBND4J_CUDALIMITTYPE_H
#define LIBND4J_CUDALIMITTYPE_H


#ifndef __JAVACPP_HACK__
enum CudaLimitType {
  CUDA_LIMIT_STACK_SIZE = 0,
  CUDA_LIMIT_MALLOC_HEAP_SIZE = 1,
  CUDA_LIMIT_PRINTF_FIFO_SIZE = 2,
  CUDA_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3,
  CUDA_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4,
  CUDA_LIMIT_MAX_L2_FETCH_GRANULARITY = 5,
  CUDA_LIMIT_PERSISTING_L2_CACHE_SIZE = 6
};
#endif
#endif  // LIBND4J_CUDALIMITTYPE_H
