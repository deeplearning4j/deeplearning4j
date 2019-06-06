/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 06.02.2019
//

#ifndef __CUDABLAS__

#include <PointersManager.h>
#include <exceptions/cuda_exception.h>
#include <logger.h>
#include <memory/Workspace.h>

namespace nd4j {

//////////////////////////////////////////////////////////////////////////
PointersManager::PointersManager(const nd4j::LaunchContext *context, const std::string& funcName)  {
    _context  = const_cast<nd4j::LaunchContext*>(context);
    _funcName = funcName;
}

//////////////////////////////////////////////////////////////////////////
void* PointersManager::replicatePointer(const void* src, const size_t numberOfBytes) {
    // no-op
    return const_cast<void *>(src);
}

//////////////////////////////////////////////////////////////////////////
void PointersManager::synchronize() const {
        // no-op
}

//////////////////////////////////////////////////////////////////////////
PointersManager::~PointersManager() {
        // no-op
}

}

#endif
