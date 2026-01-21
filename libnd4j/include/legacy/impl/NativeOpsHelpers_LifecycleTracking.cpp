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
// Implementation of lifecycle tracking native API functions
// This file contains the crash handler code and functrace-specific functions.
// Other lifecycle tracking functions are split into separate files:
// - NativeOpsHelpers_LifecycleTracking_Stats.cpp
// - NativeOpsHelpers_LifecycleTracking_Enable.cpp
// - NativeOpsHelpers_LifecycleTracking_Cache.cpp
// - NativeOpsHelpers_LifecycleTracking_Snapshot.cpp
//

#include <legacy/NativeOps.h>

// Forward declare the ComprehensiveLeakAnalyzer class before including tracker headers
namespace sd {
namespace analysis {
    class ComprehensiveLeakAnalyzer;
}
}

#include <array/NDArrayLifecycleTracker.h>
#include <array/DataBufferLifecycleTracker.h>
#include <array/TADCacheLifecycleTracker.h>
#include <array/ShapeCacheLifecycleTracker.h>
#include <array/DeallocatorServiceLifecycleTracker.h>
#include <graph/OpContextLifecycleTracker.h>
#include <ops/declarable/OpExecutionLogger.h>
#include <cstring>
#include <atomic>
#include <cstdlib>
#include <thread>
#include <csignal>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iostream>
#include <iomanip>

#ifndef _WIN32
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#endif

#ifdef __linux__
#include <sys/syscall.h>
#endif

#if defined(SD_GCC_FUNCTRACE)
#include <filesystem>
#include <array>
#include <vector>
#include <array/AllocationLogger.h>
#endif

using namespace sd::array;

// ═══════════════════════════════════════════════════════════════════════════
// CRASH HANDLER CODE - Only active when SD_GCC_FUNCTRACE is defined
// ═══════════════════════════════════════════════════════════════════════════

#if defined(SD_GCC_FUNCTRACE)
namespace {

#ifndef _WIN32
struct CrashEvent {
    int signal;
    void* faultAddress;
    long crashingThreadId;
};

constexpr int kCrashSignals[] = { SIGSEGV, SIGBUS, SIGILL, SIGFPE, SIGABRT };

class LifecycleCrashHandler {
public:
    static LifecycleCrashHandler& instance() {
        static LifecycleCrashHandler handler;
        return handler;
    }

    void ensureInitialized() {
        bool expected = false;
        if (!_initialized.compare_exchange_strong(expected, true)) {
            return;
        }

        if (::pipe(_signalPipe) != 0) {
            std::cerr << "[sd-crash] Failed to create crash notification pipe" << std::endl;
            _initialized.store(false);
            return;
        }

        _worker = std::thread(&LifecycleCrashHandler::workerLoop, this);
        _worker.detach();

        setupAltStack();
        installHandlers();
        _ready.store(true, std::memory_order_release);
    }

private:
    LifecycleCrashHandler() {
        _signalPipe[0] = -1;
        _signalPipe[1] = -1;
        _dumpComplete.store(true);
    }

    void setupAltStack() {
        const size_t altStackSize = determineAltStackSize();
        _altStackStorage.assign(altStackSize, 0);
        stack_t ss;
        ss.ss_sp = _altStackStorage.data();
        ss.ss_size = _altStackStorage.size();
        ss.ss_flags = 0;
        if (sigaltstack(&ss, &_previousAltStack) == 0) {
            _altStackInstalled = true;
        }
    }

    void installHandlers() {
        struct sigaction sa;
        std::memset(&sa, 0, sizeof(sa));
        sa.sa_sigaction = &LifecycleCrashHandler::signalHandler;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_SIGINFO | SA_ONSTACK | SA_NODEFER;

        for (size_t i = 0; i < kSignalCount; ++i) {
            if (sigaction(kCrashSignals[i], &sa, &_oldHandlers[i]) != 0) {
                std::cerr << "[sd-crash] Failed to install handler for signal "
                          << kCrashSignals[i] << std::endl;
            }
        }
        _handlersInstalled = true;
    }

    static void signalHandler(int signo, siginfo_t* info, void* ucontext) {
        LifecycleCrashHandler::instance().handleSignal(signo, info, ucontext);
    }

    void handleSignal(int signo, siginfo_t* info, void* ucontext) {
        if (!_ready.load(std::memory_order_acquire)) {
            restoreAndReraise(signo, info, ucontext);
            return;
        }

        if (_handling.exchange(true, std::memory_order_acq_rel)) {
            restoreAndReraise(signo, info, ucontext);
            return;
        }

        CrashEvent event{};
        event.signal = signo;
        event.faultAddress = info ? info->si_addr : nullptr;
        event.crashingThreadId = currentThreadId();

        ssize_t wrote = ::write(_signalPipe[1], &event, sizeof(event));
        if (wrote != sizeof(event)) {
            std::cerr << "[sd-crash] Failed to notify dump worker thread\n";
        }

        _handling.store(false, std::memory_order_release);
        restoreAndReraise(signo, info, ucontext);
    }

    void restoreAndReraise(int signo, siginfo_t* info, void* ucontext) {
        if (!_handlersInstalled) {
            ::raise(signo);
            return;
        }

        for (size_t i = 0; i < kSignalCount; ++i) {
            if (kCrashSignals[i] == signo) {
                struct sigaction& oldHandler = _oldHandlers[i];

                if (oldHandler.sa_flags & SA_SIGINFO) {
                    if (oldHandler.sa_sigaction != nullptr &&
                        oldHandler.sa_sigaction != (void (*)(int, siginfo_t*, void*))SIG_DFL &&
                        oldHandler.sa_sigaction != (void (*)(int, siginfo_t*, void*))SIG_IGN) {
                        oldHandler.sa_sigaction(signo, info, ucontext);
                        return;
                    }
                } else {
                    if (oldHandler.sa_handler != SIG_DFL && oldHandler.sa_handler != SIG_IGN) {
                        oldHandler.sa_handler(signo);
                        return;
                    }
                }

                sigaction(signo, &oldHandler, nullptr);
                ::raise(signo);
                return;
            }
        }

        ::raise(signo);
    }

    void workerLoop() {
        while (true) {
            CrashEvent event{};
            ssize_t rd = ::read(_signalPipe[0], &event, sizeof(event));
            if (rd == sizeof(event)) {
                dumpCrash(event);
                _dumpComplete.store(true, std::memory_order_release);
            }
        }
    }

    void dumpCrash(const CrashEvent &event) {
        std::string path = buildCrashFilePath();
        std::ofstream out(path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            std::cerr << "[sd-crash] Failed to open crash log at " << path << std::endl;
            return;
        }

        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm time_buf;
        localtime_r(&now_time, &time_buf);

        out << "============================================\n";
        out << "  ND4J Native Crash Report\n";
        out << "============================================\n";
        out << "Timestamp: " << std::put_time(&time_buf, "%Y-%m-%d %H:%M:%S") << "\n";
        out << "PID:       " << getpid() << "\n";
        out << "Thread:    " << event.crashingThreadId << "\n";
        out << "Signal:    " << event.signal << " (" << signalName(event.signal) << ")\n";
        out << "Address:   " << event.faultAddress << "\n\n";

        bool matched = false;

        matched |= NDArrayLifecycleTracker::getInstance().logAllocationForPointer(event.faultAddress, out);
        matched |= DataBufferLifecycleTracker::getInstance().logAllocationForAddress(event.faultAddress, out);
        matched |= ShapeCacheLifecycleTracker::getInstance().logShapeForAddress(event.faultAddress, out);
        matched |= TADCacheLifecycleTracker::getInstance().logTADForAddress(event.faultAddress, out);

        if (!matched) {
            out << "No tracked allocation matched the faulting address.\n";
        }

        out << "\n=== NDArray Snapshot ===\n";
        NDArrayLifecycleTracker::getInstance().printStatistics(out);
        NDArrayLifecycleTracker::getInstance().printCurrentLeaks(out);

        out << "\n=== DataBuffer Snapshot ===\n";
        DataBufferLifecycleTracker::getInstance().printStatistics(out);
        DataBufferLifecycleTracker::getInstance().printCurrentLeaks(out);

        out << "\n=== Shape Cache Snapshot ===\n";
        ShapeCacheLifecycleTracker::getInstance().printStatistics(out);
        ShapeCacheLifecycleTracker::getInstance().printCurrentLeaks(out);

        out << "\n=== TAD Cache Snapshot ===\n";
        TADCacheLifecycleTracker::getInstance().printStatistics(out);
        TADCacheLifecycleTracker::getInstance().printCurrentLeaks(out);

        out.close();
        std::cerr << "[sd-crash] Crash dump written to " << path << std::endl;
    }

    std::string buildCrashFilePath() {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::path cwd = fs::current_path(ec);
        if (ec) {
            cwd = ".";
        }

        std::string base = "sd_crash_pid" + std::to_string(getpid());
        fs::path candidate = cwd / (base + ".log");
        int suffix = 1;
        while (fs::exists(candidate, ec)) {
            candidate = cwd / (base + "_" + std::to_string(suffix++) + ".log");
        }
        return candidate.string();
    }

    static const char* signalName(int signo) {
        switch (signo) {
            case SIGSEGV: return "SIGSEGV";
            case SIGBUS:  return "SIGBUS";
            case SIGILL:  return "SIGILL";
            case SIGFPE:  return "SIGFPE";
            case SIGABRT: return "SIGABRT";
            default:      return "UNKNOWN";
        }
    }

    static long currentThreadId() {
#if defined(__linux__)
        return static_cast<long>(::syscall(SYS_gettid));
#else
        return static_cast<long>(reinterpret_cast<intptr_t>(pthread_self()));
#endif
    }

    static constexpr size_t kSignalCount = sizeof(kCrashSignals) / sizeof(int);
    std::atomic<bool> _initialized{false};
    std::atomic<bool> _ready{false};
    std::atomic<bool> _handling{false};
    std::atomic<bool> _dumpComplete{true};
    int _signalPipe[2];
    std::thread _worker;
    std::array<struct sigaction, kSignalCount> _oldHandlers{};
    bool _handlersInstalled{false};
    stack_t _previousAltStack{};
    bool _altStackInstalled{false};
    std::vector<uint8_t> _altStackStorage;

    static size_t determineAltStackSize() {
        long baseSize = 0;
#if defined(SIGSTKSZ)
        baseSize = SIGSTKSZ;
#endif
#if defined(MINSIGSTKSZ)
        long minSize = MINSIGSTKSZ;
#else
        long minSize = 64 * 1024;
#endif

        if (baseSize < minSize) {
            baseSize = minSize;
        }
        if (baseSize <= 0) {
            baseSize = minSize;
        }

        return static_cast<size_t>(baseSize) * 4;
    }
};

#else
class LifecycleCrashHandler {
public:
    static LifecycleCrashHandler& instance() {
        static LifecycleCrashHandler handler;
        return handler;
    }
    void ensureInitialized() {}
};
#endif  // _WIN32

}  // namespace

#endif // SD_GCC_FUNCTRACE

// ═══════════════════════════════════════════════════════════════════════════
// Functions that require SD_GCC_FUNCTRACE for full functionality
// ═══════════════════════════════════════════════════════════════════════════

SD_LIB_EXPORT void initializeLifecycleCrashHandlers() {
#if defined(SD_GCC_FUNCTRACE) && !defined(_WIN32)
    LifecycleCrashHandler::instance().ensureInitialized();
#endif
}

SD_LIB_EXPORT void enableOpExecutionLogging() {
#if defined(SD_GCC_FUNCTRACE)
    sd::ops::OpExecutionLogger::getInstance().enable();
#endif
}

SD_LIB_EXPORT void disableOpExecutionLogging() {
#if defined(SD_GCC_FUNCTRACE)
    sd::ops::OpExecutionLogger::getInstance().disable();
#endif
}

SD_LIB_EXPORT bool isOpExecutionLoggingEnabled() {
#if defined(SD_GCC_FUNCTRACE)
    return sd::ops::OpExecutionLogger::getInstance().isEnabled();
#else
    return false;
#endif
}

SD_LIB_EXPORT const char* getOpExecutionLogPath() {
#if defined(SD_GCC_FUNCTRACE)
    static thread_local std::string g_opLogPath;
    g_opLogPath = sd::ops::OpExecutionLogger::getInstance().getLogPath();
    return g_opLogPath.c_str();
#else
    static const char* empty = "";
    return empty;
#endif
}

SD_LIB_EXPORT const char* getOpExecutionLogContents(size_t maxBytes, bool fromEnd) {
#if defined(SD_GCC_FUNCTRACE)
    static thread_local std::string g_opLogContents;
    g_opLogContents = sd::ops::OpExecutionLogger::getInstance().getLogContents(maxBytes, fromEnd);
    return g_opLogContents.c_str();
#else
    static const char* empty = "";
    return empty;
#endif
}

SD_LIB_EXPORT void dumpOpExecutionLog() {
#if defined(SD_GCC_FUNCTRACE)
    sd::ops::OpExecutionLogger::getInstance().flush();
#endif
}

SD_LIB_EXPORT void dumpOpExecutionState(const char* message) {
#if defined(SD_GCC_FUNCTRACE)
    std::string msg = message ? message : "";
    sd::ops::OpExecutionLogger::getInstance().dumpCurrentState(msg);
#endif
}

SD_LIB_EXPORT const char* getAllocationLogPath() {
#if defined(SD_GCC_FUNCTRACE)
    static thread_local std::string g_allocLogPath;
    g_allocLogPath = sd::array::AllocationLogger::getInstance().getLogPath();
    return g_allocLogPath.c_str();
#else
    static const char* empty = "";
    return empty;
#endif
}

SD_LIB_EXPORT void updateAllocationJavaStackTrace(OpaqueNDArray array, const char* javaStackTrace) {
#if defined(SD_GCC_FUNCTRACE)
    if (array != nullptr && javaStackTrace != nullptr) {
        NDArrayLifecycleTracker::getInstance().updateJavaStackTrace(array, std::string(javaStackTrace));
    }
#endif
}
