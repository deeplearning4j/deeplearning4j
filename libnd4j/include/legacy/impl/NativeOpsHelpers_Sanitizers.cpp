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
 * Triggers LeakSanitizer to perform a leak check immediately.
 * This allows checking for leaks at any point during execution,
 * not just at program exit.
 *
 * CRITICAL: Clears TAD and Shape caches BEFORE checking for leaks
 * to prevent false positives from legitimate cached data.
 *
 * Cleanup sequence (matches MainApplication.java shutdown handler):
 * 1. Clear TAD cache (frees cached TadPack objects)
 * 2. Clear Shape cache (frees cached shape info)
 * 3. Trigger leak check
 *
 * When built without sanitizers, this is a no-op.
 * Safe to call from Java via JNI.
 */
SD_LIB_EXPORT void triggerLeakCheck() {
#ifdef HAS_LEAK_SANITIZER
    // Clear caches before leak check to prevent false positives
    // These caches contain legitimate data that should not be reported as leaks
    clearTADCache();
    clearShapeCache();

    // Now check for actual leaks
    __lsan_do_leak_check();
#else
    // No-op when not built with ASAN/LSAN
#endif
}
