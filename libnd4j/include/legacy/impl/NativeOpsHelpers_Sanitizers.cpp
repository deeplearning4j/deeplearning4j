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
// Sanitizer utilities - platform independent
//

#include <legacy/NativeOps.h>

// LSAN leak check trigger - only available when built with ASAN/LSAN
#if defined(__has_feature)
  #if __has_feature(address_sanitizer)
    extern "C" void __lsan_do_leak_check(void);
    #define HAS_LEAK_SANITIZER 1
  #endif
#elif defined(__SANITIZE_ADDRESS__)
  // GCC doesn't have __has_feature, but defines __SANITIZE_ADDRESS__
  extern "C" void __lsan_do_leak_check(void);
  #define HAS_LEAK_SANITIZER 1
#endif

// MSAN doesn't have leak detection - it only tracks uninitialized memory
// If built with MSAN, this will be a no-op

/**
 * Triggers leak checking and clears caches.
 * This allows checking for leaks at any point during execution,
 * not just at program exit.
 *
 * Always clears TAD and Shape caches before checking for leaks
 * to prevent false positives from legitimate cached data.
 *
 * Cleanup sequence (matches MainApplication.java shutdown handler):
 * 1. Clear TAD cache (frees cached TadPack objects)
 * 2. Clear Shape cache (frees cached shape info)
 * 3. Trigger leak check (if sanitizers are enabled)
 *
 * Safe to call from Java via JNI.
 */
SD_LIB_EXPORT void triggerLeakCheck() {
    // not just when HAS_LEAK_SANITIZER is defined.
    //
    // WHY: Custom lifecycle tracking (TADCacheLifecycleTracker) is used
    // even when building with MSan (Memory Sanitizer), which doesn't define
    // HAS_LEAK_SANITIZER. But lifecycle tracking still needs caches cleared
    // before reporting to avoid false positives.
    //
    // TAD and Shape caches contain legitimate data structures that persist
    // across operations for performance. They are NOT memory leaks.
    clearTADCache();
    clearShapeCache();

#ifdef HAS_LEAK_SANITIZER
    // Additionally trigger sanitizer leak check if available
    __lsan_do_leak_check();
#else
    // No sanitizer leak check, but cache clearing still happened above
    // This ensures custom lifecycle tracking doesn't report false positives
#endif
}
