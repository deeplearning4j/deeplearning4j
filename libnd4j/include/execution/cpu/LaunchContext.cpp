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
// Created by raver119 on 30.11.17.
//

#include <execution/LaunchContext.h>
#include <execution/AffinityManager.h>
#include <helpers/logger.h>
#include <exceptions/cuda_exception.h>
#include <thread>

#if defined(SD_IOS_BUILD) || defined(SD_APPLE_BUILD) || defined(SD_ANDROID_BUILD)
sd::ContextBuffers contextBuffers = sd::ContextBuffers();
#else
thread_local sd::ContextBuffers contextBuffers = sd::ContextBuffers();
#endif

#ifdef HAVE_MKLDNN
#include <dnnl.hpp>
#endif

namespace sd {

    LaunchContext::~LaunchContext() {
#ifdef HAVE_MKLDNN
        delete reinterpret_cast<dnnl::engine*>(_engine);
#endif
    }

    std::vector<std::shared_ptr<LaunchContext>> LaunchContext::_contexts = std::vector<std::shared_ptr<LaunchContext>>();
    MAP_IMPL<int, std::mutex*> LaunchContext::_deviceMutexes;
    std::mutex LaunchContext::_mutex;

////////////////////////////////////////////////////////////////////////
    LaunchContext::LaunchContext() {
        // default constructor, just to make clang/ranlib happy
        _workspace = nullptr;
        _deviceID = 0;

#ifdef HAVE_MKLDNN
        _engine = new dnnl::engine(dnnl::engine::kind::cpu, 0);
#endif
    }

    LaunchContext::LaunchContext(Nd4jPointer cudaStream, Nd4jPointer reductionPointer, Nd4jPointer scalarPointer, Nd4jPointer allocationPointer) {

    }

    static std::mutex _lock;

    LaunchContext* LaunchContext::defaultContext() {
      {
        // synchronous block goes here
        std::lock_guard<std::mutex> lock(_lock);
        // TODO: we need it to be device-aware, but only once we add NUMA support for cpu
        if (LaunchContext::_contexts.empty())
          LaunchContext::_contexts.emplace_back(std::make_shared<LaunchContext>());
      }

      // return context for current device
      return LaunchContext::_contexts[0].get();
    }

    std::mutex* LaunchContext::deviceMutex() {
        return &_mutex;
    }

    void LaunchContext::swapContextBuffers(ContextBuffers &buffers) {
        //
    }

    bool LaunchContext::isInitialized() {
        return true;
    }

    void LaunchContext::releaseBuffers() {
        //
    }

    sd::ErrorReference* LaunchContext::errorReference() {
        return contextBuffers.errorReference();
    }

    void* LaunchContext::engine() {
        return _engine;
    }
}