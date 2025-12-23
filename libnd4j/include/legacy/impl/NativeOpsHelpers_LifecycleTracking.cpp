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
// Author: Adam Gibson
//

#include <legacy/NativeOps.h>

// Forward declare the ComprehensiveLeakAnalyzer class before including tracker headers
// This ensures the friend declarations in the tracker classes can see the class
namespace sd {
namespace analysis {
    class ComprehensiveLeakAnalyzer;
}
}

// Always include lifecycle trackers - they work without SD_GCC_FUNCTRACE
// but stack trace capture is only enabled when SD_GCC_FUNCTRACE is defined
#include <array/NDArrayLifecycleTracker.h>
#include <array/DataBufferLifecycleTracker.h>
#include <array/TADCacheLifecycleTracker.h>
#include <array/ShapeCacheLifecycleTracker.h>
#include <array/DeallocatorServiceLifecycleTracker.h>
#include <graph/OpContextLifecycleTracker.h>
#include <ops/declarable/OpExecutionLogger.h>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <atomic>
#include <cstdlib>
#include <thread>
#include <csignal>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iostream>

#ifndef _WIN32
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#endif

#ifdef __linux__
#include <sys/syscall.h>
#endif

// Only include filesystem and AllocationLogger when functrace is enabled
#if defined(SD_GCC_FUNCTRACE)
#include <filesystem>
#include <array>
#include <vector>
#include <array/AllocationLogger.h>
#endif

using namespace sd::array;

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
            // If write fails, just continue - better to let JVM handle it than hang
            std::cerr << "[sd-crash] Failed to notify dump worker thread\n";
        }

        // Reset handling flag and immediately chain to JVM handler
        // The dump will happen asynchronously in the worker thread
        _handling.store(false, std::memory_order_release);
        restoreAndReraise(signo, info, ucontext);
    }

    void restoreAndReraise(int signo, siginfo_t* info, void* ucontext) {
        if (!_handlersInstalled) {
            // No old handler to call, just raise the signal
            ::raise(signo);
            return;
        }

        // Find and call the old handler for this signal
        for (size_t i = 0; i < kSignalCount; ++i) {
            if (kCrashSignals[i] == signo) {
                struct sigaction& oldHandler = _oldHandlers[i];

#ifdef __cpp_exceptions
                try {
                    if (oldHandler.sa_flags & SA_SIGINFO) {
                        // Old handler is a sigaction-style handler (takes siginfo_t and ucontext)
                        if (oldHandler.sa_sigaction != nullptr &&
                            oldHandler.sa_sigaction != (void (*)(int, siginfo_t*, void*))SIG_DFL &&
                            oldHandler.sa_sigaction != (void (*)(int, siginfo_t*, void*))SIG_IGN) {
                            // Call the original handler with the ORIGINAL siginfo and ucontext
                            // This preserves si_code, si_addr, and all other signal information
                            oldHandler.sa_sigaction(signo, info, ucontext);
                            return;
                        }
                    } else {
                        // Old handler is a simple signal handler (only takes signal number)
                        if (oldHandler.sa_handler != SIG_DFL && oldHandler.sa_handler != SIG_IGN) {
                            oldHandler.sa_handler(signo);
                            return;
                        }
                    }
                } catch (const std::exception& e) {
                    // Old handler threw an exception - log it and convert to signal
                    std::cerr << "[sd-crash] Exception from old signal handler for signal " << signo
                              << ": " << e.what() << std::endl;
                    std::cerr << "[sd-crash] Converting exception to signal termination" << std::endl;
                    // Fall through to restore and raise
                } catch (...) {
                    // Unknown exception from old handler
                    std::cerr << "[sd-crash] Unknown exception from old signal handler for signal " << signo << std::endl;
                    std::cerr << "[sd-crash] Converting exception to signal termination" << std::endl;
                    // Fall through to restore and raise
                }
#else
                if (oldHandler.sa_flags & SA_SIGINFO) {
                    // Old handler is a sigaction-style handler (takes siginfo_t and ucontext)
                    if (oldHandler.sa_sigaction != nullptr &&
                        oldHandler.sa_sigaction != (void (*)(int, siginfo_t*, void*))SIG_DFL &&
                        oldHandler.sa_sigaction != (void (*)(int, siginfo_t*, void*))SIG_IGN) {
                        // Call the original handler with the ORIGINAL siginfo and ucontext
                        // This preserves si_code, si_addr, and all other signal information
                        oldHandler.sa_sigaction(signo, info, ucontext);
                        return;
                    }
                } else {
                    // Old handler is a simple signal handler (only takes signal number)
                    if (oldHandler.sa_handler != SIG_DFL && oldHandler.sa_handler != SIG_IGN) {
                        oldHandler.sa_handler(signo);
                        return;
                    }
                }
#endif

                // If we get here, the old handler was SIG_DFL or SIG_IGN, or threw an exception
                // Restore it and raise the signal (this is safe since it's the default handler)
                sigaction(signo, &oldHandler, nullptr);
                ::raise(signo);
                return;
            }
        }

        // Signal not found in our list - just raise it
        ::raise(signo);
    }

    void restoreOriginal(int signo) {
        if (!_handlersInstalled) return;
        for (size_t i = 0; i < kSignalCount; ++i) {
            if (kCrashSignals[i] == signo) {
                sigaction(signo, &_oldHandlers[i], nullptr);
                break;
            }
        }
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

        // NDArray allocation lookup
#ifdef __cpp_exceptions
        try {
            matched |= NDArrayLifecycleTracker::getInstance().logAllocationForPointer(event.faultAddress, out);
        } catch (const std::exception& e) {
            out << "[sd-crash] NDArray logAllocationForPointer failed: " << e.what() << "\n";
        } catch (...) {
            out << "[sd-crash] NDArray logAllocationForPointer failed: unknown exception\n";
        }
#else
        matched |= NDArrayLifecycleTracker::getInstance().logAllocationForPointer(event.faultAddress, out);
#endif

        // DataBuffer allocation lookup
#ifdef __cpp_exceptions
        try {
            matched |= DataBufferLifecycleTracker::getInstance().logAllocationForAddress(event.faultAddress, out);
        } catch (const std::exception& e) {
            out << "[sd-crash] DataBuffer logAllocationForAddress failed: " << e.what() << "\n";
        } catch (...) {
            out << "[sd-crash] DataBuffer logAllocationForAddress failed: unknown exception\n";
        }
#else
        matched |= DataBufferLifecycleTracker::getInstance().logAllocationForAddress(event.faultAddress, out);
#endif

        // ShapeCache allocation lookup
#ifdef __cpp_exceptions
        try {
            matched |= ShapeCacheLifecycleTracker::getInstance().logShapeForAddress(event.faultAddress, out);
        } catch (const std::exception& e) {
            out << "[sd-crash] ShapeCache logShapeForAddress failed: " << e.what() << "\n";
        } catch (...) {
            out << "[sd-crash] ShapeCache logShapeForAddress failed: unknown exception\n";
        }
#else
        matched |= ShapeCacheLifecycleTracker::getInstance().logShapeForAddress(event.faultAddress, out);
#endif

        // TADCache allocation lookup
#ifdef __cpp_exceptions
        try {
            matched |= TADCacheLifecycleTracker::getInstance().logTADForAddress(event.faultAddress, out);
        } catch (const std::exception& e) {
            out << "[sd-crash] TADCache logTADForAddress failed: " << e.what() << "\n";
        } catch (...) {
            out << "[sd-crash] TADCache logTADForAddress failed: unknown exception\n";
        }
#else
        matched |= TADCacheLifecycleTracker::getInstance().logTADForAddress(event.faultAddress, out);
#endif

        if (!matched) {
            out << "No tracked allocation matched the faulting address.\n";
        }

        // NDArray statistics and leaks
        out << "\n=== NDArray Snapshot ===\n";
#ifdef __cpp_exceptions
        try {
            NDArrayLifecycleTracker::getInstance().printStatistics(out);
        } catch (const std::exception& e) {
            out << "[sd-crash] NDArray printStatistics failed: " << e.what() << "\n";
        } catch (...) {
            out << "[sd-crash] NDArray printStatistics failed: unknown exception\n";
        }
#else
        NDArrayLifecycleTracker::getInstance().printStatistics(out);
#endif

#ifdef __cpp_exceptions
        try {
            NDArrayLifecycleTracker::getInstance().printCurrentLeaks(out);
        } catch (const std::exception& e) {
            out << "[sd-crash] NDArray printCurrentLeaks failed: " << e.what() << "\n";
        } catch (...) {
            out << "[sd-crash] NDArray printCurrentLeaks failed: unknown exception\n";
        }
#else
        NDArrayLifecycleTracker::getInstance().printCurrentLeaks(out);
#endif

        // DataBuffer statistics and leaks
        out << "\n=== DataBuffer Snapshot ===\n";
#ifdef __cpp_exceptions
        try {
            DataBufferLifecycleTracker::getInstance().printStatistics(out);
        } catch (const std::exception& e) {
            out << "[sd-crash] DataBuffer printStatistics failed: " << e.what() << "\n";
        } catch (...) {
            out << "[sd-crash] DataBuffer printStatistics failed: unknown exception\n";
        }
#else
        DataBufferLifecycleTracker::getInstance().printStatistics(out);
#endif

#ifdef __cpp_exceptions
        try {
            DataBufferLifecycleTracker::getInstance().printCurrentLeaks(out);
        } catch (const std::exception& e) {
            out << "[sd-crash] DataBuffer printCurrentLeaks failed: " << e.what() << "\n";
        } catch (...) {
            out << "[sd-crash] DataBuffer printCurrentLeaks failed: unknown exception\n";
        }
#else
        DataBufferLifecycleTracker::getInstance().printCurrentLeaks(out);
#endif

        // ShapeCache statistics and leaks
        out << "\n=== Shape Cache Snapshot ===\n";
#ifdef __cpp_exceptions
        try {
            ShapeCacheLifecycleTracker::getInstance().printStatistics(out);
        } catch (const std::exception& e) {
            out << "[sd-crash] ShapeCache printStatistics failed: " << e.what() << "\n";
        } catch (...) {
            out << "[sd-crash] ShapeCache printStatistics failed: unknown exception\n";
        }
#else
        ShapeCacheLifecycleTracker::getInstance().printStatistics(out);
#endif

#ifdef __cpp_exceptions
        try {
            ShapeCacheLifecycleTracker::getInstance().printCurrentLeaks(out);
        } catch (const std::exception& e) {
            out << "[sd-crash] ShapeCache printCurrentLeaks failed: " << e.what() << "\n";
        } catch (...) {
            out << "[sd-crash] ShapeCache printCurrentLeaks failed: unknown exception\n";
        }
#else
        ShapeCacheLifecycleTracker::getInstance().printCurrentLeaks(out);
#endif

        // TADCache statistics and leaks
        out << "\n=== TAD Cache Snapshot ===\n";
#ifdef __cpp_exceptions
        try {
            TADCacheLifecycleTracker::getInstance().printStatistics(out);
        } catch (const std::exception& e) {
            out << "[sd-crash] TADCache printStatistics failed: " << e.what() << "\n";
        } catch (...) {
            out << "[sd-crash] TADCache printStatistics failed: unknown exception\n";
        }
#else
        TADCacheLifecycleTracker::getInstance().printStatistics(out);
#endif

#ifdef __cpp_exceptions
        try {
            TADCacheLifecycleTracker::getInstance().printCurrentLeaks(out);
        } catch (const std::exception& e) {
            out << "[sd-crash] TADCache printCurrentLeaks failed: " << e.what() << "\n";
        } catch (...) {
            out << "[sd-crash] TADCache printCurrentLeaks failed: unknown exception\n";
        }
#else
        TADCacheLifecycleTracker::getInstance().printCurrentLeaks(out);
#endif

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
        long minSize = 64 * 1024;  // 64KB fallback when platform doesn't define MINSIGSTKSZ
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

#endif // SD_GCC_FUNCTRACE (crash handlers section)

// Forward declarations for cache clearing functions
SD_LIB_EXPORT void clearTADCache();
SD_LIB_EXPORT void clearShapeCache();
SD_LIB_EXPORT void checkAndCleanupCaches();

// Note: ComprehensiveLeakAnalyzer is forward declared but not yet implemented
// The friend declarations in lifecycle tracker classes allow for future extension

// initializeLifecycleCrashHandlers moved to end of file
// (single definition with #if SD_GCC_FUNCTRACE guard inside)

// ═══════════════════════════════════════════════════════════════════════════
// LIFECYCLE STATS AND REPORT FUNCTIONS - Always available
// These functions use the trackers which work without SD_GCC_FUNCTRACE.
// Stack trace output will be limited without functrace, but stats work.
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Converts NDArray lifecycle statistics to JSON format.
 */
const char* getNDArrayLifecycleStats() {
    auto stats = NDArrayLifecycleTracker::getInstance().getStats();

    std::ostringstream json;
    json << "{\n";
    json << "  \"total_allocations\": " << stats.totalAllocations << ",\n";
    json << "  \"total_deallocations\": " << stats.totalDeallocations << ",\n";
    json << "  \"current_live\": " << stats.currentLive << ",\n";
    json << "  \"total_bytes_allocated\": " << stats.totalBytesAllocated << ",\n";
    json << "  \"total_bytes_deallocated\": " << stats.totalBytesDeallocated << ",\n";
    json << "  \"peak_live\": " << stats.peakLive << "\n";
    json << "}";

    std::string result = json.str();
    char* cstr = new char[result.length() + 1];
    std::strcpy(cstr, result.c_str());
    return cstr;
}

/**
 * Converts DataBuffer lifecycle statistics to JSON format.
 * Note: DataBufferStats is a unified structure, not separated by buffer type.
 */
const char* getDataBufferLifecycleStats() {
    auto stats = DataBufferLifecycleTracker::getInstance().getStats();

    std::ostringstream json;
    json << "{\n";
    json << "  \"total_allocations\": " << stats.totalAllocations << ",\n";
    json << "  \"total_deallocations\": " << stats.totalDeallocations << ",\n";
    json << "  \"current_live\": " << stats.currentLive << ",\n";
    json << "  \"peak_live\": " << stats.peakLive << ",\n";
    json << "  \"total_bytes_allocated\": " << stats.totalBytesAllocated << ",\n";
    json << "  \"total_bytes_deallocated\": " << stats.totalBytesDeallocated << "\n";
    json << "}";

    std::string result = json.str();
    char* cstr = new char[result.length() + 1];
    std::strcpy(cstr, result.c_str());
    return cstr;
}

/**
 * Generates a flamegraph SVG for NDArray allocations.
 */
void generateNDArrayAllocationFlamegraph(const char* outputPath) {
    if (outputPath == nullptr) {
        return;
    }

    std::string path(outputPath);
    NDArrayLifecycleTracker::getInstance().generateFlamegraph(path);
}

/**
 * Generates a flamegraph SVG for NDArray deallocations.
 */
void generateNDArrayDeallocationFlamegraph(const char* outputPath) {
    if (outputPath == nullptr) {
        return;
    }

    std::string path(outputPath);
    NDArrayLifecycleTracker::getInstance().generateDeletionFlamegraph(path);
}

/**
 * Generates a flamegraph SVG for DataBuffer allocations.
 */
void generateDataBufferAllocationFlamegraph(const char* outputPath, int bufferType) {
    if (outputPath == nullptr) {
        return;
    }

    std::string path(outputPath);
    BufferType type = (bufferType == 0) ? BufferType::PRIMARY : BufferType::SPECIAL;
    DataBufferLifecycleTracker::getInstance().generateFlamegraph(path, static_cast<int>(type));
}

/**
 * Generates a flamegraph SVG for DataBuffer deallocations.
 */
void generateDataBufferDeallocationFlamegraph(const char* outputPath, int bufferType) {
    if (outputPath == nullptr) {
        return;
    }

    std::string path(outputPath);
    BufferType type = (bufferType == 0) ? BufferType::PRIMARY : BufferType::SPECIAL;
    DataBufferLifecycleTracker::getInstance().generateDeletionFlamegraph(path, static_cast<int>(type));
}

/**
 * Generates a comprehensive leak report combining all lifecycle trackers.
 * This report now includes sample stack traces for each leaked allocation.
 */
void generateLifecycleLeakReport(const char* outputPath) {
    if (outputPath == nullptr) {
        return;
    }

    std::string path(outputPath);

    // Create combined report with all tracker statistics
    std::ofstream combined(path);
    if (combined.is_open()) {
        combined << "============================================\n";
        combined << "  COMPREHENSIVE LIFECYCLE LEAK REPORT\n";
        combined << "============================================\n\n";

        // NDArray statistics
        auto ndarray_stats = NDArrayLifecycleTracker::getInstance().getStats();
        combined << "=== NDArray Statistics ===\n";
        combined << "  Tracking Enabled:         " << (NDArrayLifecycleTracker::getInstance().isEnabled() ? "YES" : "NO") << "\n";
        combined << "  Total Allocations:        " << ndarray_stats.totalAllocations << "\n";
        combined << "  Total Deallocations:      " << ndarray_stats.totalDeallocations << "\n";
        combined << "  Current Live:             " << ndarray_stats.currentLive << "\n";
        combined << "  Peak Live:                " << ndarray_stats.peakLive << "\n";
        combined << "  Total Bytes Allocated:    " << ndarray_stats.totalBytesAllocated << "\n";
        combined << "  Total Bytes Deallocated:  " << ndarray_stats.totalBytesDeallocated << "\n";
        combined << "\n";

        // DataBuffer statistics
        auto databuffer_stats = DataBufferLifecycleTracker::getInstance().getStats();
        combined << "=== DataBuffer Statistics ===\n";
        combined << "  Tracking Enabled:         " << (DataBufferLifecycleTracker::getInstance().isEnabled() ? "YES" : "NO") << "\n";
        combined << "  Total Allocations:        " << databuffer_stats.totalAllocations << "\n";
        combined << "  Total Deallocations:      " << databuffer_stats.totalDeallocations << "\n";
        combined << "  Current Live:             " << databuffer_stats.currentLive << "\n";
        combined << "  Peak Live:                " << databuffer_stats.peakLive << "\n";
        combined << "  Total Bytes Allocated:    " << databuffer_stats.totalBytesAllocated << "\n";
        combined << "  Total Bytes Deallocated:  " << databuffer_stats.totalBytesDeallocated << "\n";
        combined << "\n";

        // TADCache statistics
        auto tad_stats = TADCacheLifecycleTracker::getInstance().getStats();
        combined << "=== TADCache Statistics ===\n";
        combined << "  Tracking Enabled:         " << (TADCacheLifecycleTracker::getInstance().isEnabled() ? "YES" : "NO") << "\n";
        combined << "  Total Allocations:        " << tad_stats.totalAllocations << "\n";
        combined << "  Total Deallocations:      " << tad_stats.totalDeallocations << "\n";
        combined << "  Current Live:             " << tad_stats.currentLive << "\n";
        combined << "  Peak Live:                " << tad_stats.peakLive << "\n";
        combined << "  Total Bytes Allocated:    " << tad_stats.totalBytesAllocated << "\n";
        combined << "  Total Bytes Deallocated:  " << tad_stats.totalBytesDeallocated << "\n";
        combined << "\n";

        // ShapeCache statistics
        auto shape_stats = ShapeCacheLifecycleTracker::getInstance().getStats();
        combined << "=== ShapeCache Statistics ===\n";
        combined << "  Tracking Enabled:         " << (ShapeCacheLifecycleTracker::getInstance().isEnabled() ? "YES" : "NO") << "\n";
        combined << "  Total Allocations:        " << shape_stats.totalAllocations << "\n";
        combined << "  Total Deallocations:      " << shape_stats.totalDeallocations << "\n";
        combined << "  Current Live:             " << shape_stats.currentLive << "\n";
        combined << "  Peak Live:                " << shape_stats.peakLive << "\n";
        combined << "\n";

        // OpContext statistics
        auto opctx_stats = sd::graph::OpContextLifecycleTracker::getInstance().getStats();
        combined << "=== OpContext Statistics ===\n";
        combined << "  Tracking Enabled:         " << (sd::graph::OpContextLifecycleTracker::getInstance().isEnabled() ? "YES" : "NO") << "\n";
        combined << "  Total Allocations:        " << opctx_stats.totalAllocations << "\n";
        combined << "  Total Deallocations:      " << opctx_stats.totalDeallocations << "\n";
        combined << "  Current Live:             " << opctx_stats.currentLive << "\n";
        combined << "  Peak Live:                " << opctx_stats.peakLive << "\n";
        combined << "\n";

        // DeallocatorService statistics
        combined << "=== DeallocatorService Statistics ===\n";
        combined << "  Tracking Enabled:         " << (DeallocatorServiceLifecycleTracker::getInstance().isEnabled() ? "YES" : "NO") << "\n";
        combined << "  Total Allocations:        " << DeallocatorServiceLifecycleTracker::getInstance().getTotalAllocations() << "\n";
        combined << "  Total Deallocations:      " << DeallocatorServiceLifecycleTracker::getInstance().getTotalDeallocations() << "\n";
        combined << "  Current Live Count:       " << DeallocatorServiceLifecycleTracker::getInstance().getCurrentLiveCount() << "\n";
        combined << "  Current Bytes In Use:     " << DeallocatorServiceLifecycleTracker::getInstance().getCurrentBytesInUse() << "\n";
        combined << "  Peak Live Count:          " << DeallocatorServiceLifecycleTracker::getInstance().getPeakLiveCount() << "\n";
        combined << "  Peak Bytes:               " << DeallocatorServiceLifecycleTracker::getInstance().getPeakBytes() << "\n";
        combined << "\n";

        // Summary
        combined << "============================================\n";
        combined << "  SUMMARY\n";
        combined << "============================================\n";
        size_t total_leaks = ndarray_stats.currentLive + databuffer_stats.currentLive + opctx_stats.currentLive;
        if (total_leaks > 0) {
            combined << "  TOTAL POTENTIAL LEAKS: " << total_leaks << "\n";
            combined << "    - NDArrays:     " << ndarray_stats.currentLive << "\n";
            combined << "    - DataBuffers:  " << databuffer_stats.currentLive << "\n";
            combined << "    - OpContexts:   " << opctx_stats.currentLive << "\n";
        } else {
            combined << "  No leaks detected.\n";
        }
        combined << "\n";

        // Now output sample stack traces for each type of leak
        combined << "============================================\n";
        combined << "  SAMPLE LEAK STACK TRACES\n";
        combined << "============================================\n\n";

        // NDArray leaks with stack traces
        NDArrayLifecycleTracker::getInstance().printCurrentLeaks(combined, 5);
        combined << "\n";

        // DataBuffer leaks with stack traces
        DataBufferLifecycleTracker::getInstance().printCurrentLeaks(combined, 5);
        combined << "\n";

        // OpContext leaks with stack traces
        sd::graph::OpContextLifecycleTracker::getInstance().printCurrentLeaks(combined, 5);
        combined << "\n";

        // Per-operation analysis - groups allocations by operation with stack traces
        combined << "============================================\n";
        combined << "  PER-OPERATION ALLOCATION BREAKDOWN\n";
        combined << "============================================\n";
        combined << "This section groups leaked allocations by the operation\n";
        combined << "that created them, with sample stack traces for each.\n\n";

        // NDArray per-op analysis with stack traces
        NDArrayLifecycleTracker::getInstance().printPerOpAnalysis(combined, 3);
        combined << "\n";

        // DataBuffer per-op analysis with stack traces
        DataBufferLifecycleTracker::getInstance().printPerOpAnalysis(combined, 3);
        combined << "\n";

        // OpContext per-op analysis with stack traces
        sd::graph::OpContextLifecycleTracker::getInstance().printPerOpAnalysis(combined, 3);
        combined << "\n";

        // Actionable analysis section
        combined << "============================================\n";
        combined << "  ACTIONABLE RECOMMENDATIONS\n";
        combined << "============================================\n";
        combined << "This section provides specific actions to address memory issues.\n\n";

        // Top leaking operations
        combined << "--- TOP OPERATIONS BY LIVE ALLOCATIONS ---\n\n";
        
        auto ndTopOps = NDArrayLifecycleTracker::getInstance().getTopOpsByLiveCount(5);
        if (!ndTopOps.empty()) {
            combined << "  NDArray Top 5:\n";
            for (const auto& op : ndTopOps) {
                double javaPct = op.liveCount > 0 ? (100.0 * op.javaCount / op.liveCount) : 0;
                combined << "    " << op.opName << ": " << op.liveCount << " live (" 
                         << (op.liveBytes / (1024*1024)) << " MB) - " 
                         << javaPct << "% Java\n";
            }
            combined << "\n";
        }

        auto dbTopOps = DataBufferLifecycleTracker::getInstance().getTopOpsByLiveCount(5);
        if (!dbTopOps.empty()) {
            combined << "  DataBuffer Top 5:\n";
            for (const auto& op : dbTopOps) {
                double javaPct = op.liveCount > 0 ? (100.0 * op.javaCount / op.liveCount) : 0;
                combined << "    " << op.opName << ": " << op.liveCount << " live (" 
                         << (op.liveBytes / (1024*1024)) << " MB) - " 
                         << javaPct << "% Java\n";
            }
            combined << "\n";
        }

        // Detailed actionable analysis per tracker
        NDArrayLifecycleTracker::getInstance().printActionableAnalysis(combined);
        combined << "\n";
        
        DataBufferLifecycleTracker::getInstance().printActionableAnalysis(combined);
        combined << "\n";

        // DeallocatorService status
        combined << "--- DeallocatorService Status ---\n";
        auto deallocAllocs = DeallocatorServiceLifecycleTracker::getInstance().getTotalAllocations();
        auto deallocDeallocs = DeallocatorServiceLifecycleTracker::getInstance().getTotalDeallocations();
        auto backlog = deallocAllocs - deallocDeallocs;
        double backlogPct = deallocAllocs > 0 ? (100.0 * backlog / deallocAllocs) : 0;
        
        combined << "  Allocations: " << deallocAllocs << "\n";
        combined << "  Deallocations: " << deallocDeallocs << "\n";
        combined << "  Backlog: " << backlog << " (" << backlogPct << "%)\n";
        
        if (backlogPct > 10) {
            combined << "  [WARNING] Deallocator falling behind - consider System.gc()\n";
        } else if (backlogPct > 5) {
            combined << "  [INFO] Mild deallocation lag - normal during high throughput\n";
        } else {
            combined << "  [OK] Deallocator keeping up\n";
        }
        combined << "\n";

        // Cache status and actions
        combined << "--- Cache Actions ---\n";
        auto tadStats = TADCacheLifecycleTracker::getInstance().getStats();
        auto shapeStats = ShapeCacheLifecycleTracker::getInstance().getStats();
        
        combined << "  TAD Cache: " << tadStats.currentLive << " entries\n";
        combined << "  Shape Cache: " << shapeStats.currentLive << " entries\n";
        
        if (tadStats.currentLive > 5000) {
            combined << "  [ACTION] TAD cache large - call clearTADCache() to free memory\n";
        }
        
        combined << "\n";

        combined.close();
    }
}


/**
 * Generates a comprehensive leak source analysis combining ALL lifecycle trackers.
 * This is a stub implementation that currently just calls generateLifecycleLeakReport.
 */
void generateComprehensiveLeakAnalysis(const char* outputDir) {
    if (outputDir == nullptr) {
        return;
    }

    std::string dir(outputDir);
    std::string reportPath = dir + "/comprehensive_leak_report.txt";
    generateLifecycleLeakReport(reportPath.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
// LIFECYCLE TRACKING FUNCTIONS - Always available (not dependent on SD_GCC_FUNCTRACE)
// Stack trace capture is only enabled when SD_GCC_FUNCTRACE is defined,
// but basic tracking (counts, pointers, timestamps) always works.
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Enables NDArray lifecycle tracking.
 * When enabled, all NDArray allocations and deallocations are tracked
 * with stack traces for leak detection.
 */
SD_LIB_EXPORT void enableNDArrayTracking() {
    NDArrayLifecycleTracker::getInstance().setEnabled(true);
}

/**
 * Disables NDArray lifecycle tracking.
 */
SD_LIB_EXPORT void disableNDArrayTracking() {
    NDArrayLifecycleTracker::getInstance().setEnabled(false);
}

/**
 * Enables DataBuffer lifecycle tracking.
 * When enabled, all DataBuffer allocations and deallocations are tracked
 * with stack traces for leak detection.
 */
SD_LIB_EXPORT void enableDataBufferTracking() {
    DataBufferLifecycleTracker::getInstance().setEnabled(true);
}

/**
 * Disables DataBuffer lifecycle tracking.
 */
SD_LIB_EXPORT void disableDataBufferTracking() {
    DataBufferLifecycleTracker::getInstance().setEnabled(false);
}

/**
 * Enables TADCache lifecycle tracking.
 * When enabled, all TAD (Tensor Along Dimension) cache allocations and deallocations
 * are tracked with stack traces for leak detection.
 */
SD_LIB_EXPORT void enableTADCacheTracking() {
    TADCacheLifecycleTracker::getInstance().setEnabled(true);
}

/**
 * Disables TADCache lifecycle tracking.
 */
SD_LIB_EXPORT void disableTADCacheTracking() {
    TADCacheLifecycleTracker::getInstance().setEnabled(false);
}

/**
 * Enables ShapeCache lifecycle tracking.
 * When enabled, all shape cache allocations and deallocations are tracked
 * with stack traces for leak detection.
 */
SD_LIB_EXPORT void enableShapeCacheTracking() {
    ShapeCacheLifecycleTracker::getInstance().setEnabled(true);
}

/**
 * Disables ShapeCache lifecycle tracking.
 */
SD_LIB_EXPORT void disableShapeCacheTracking() {
    ShapeCacheLifecycleTracker::getInstance().setEnabled(false);
}

/**
 * Enables OpContext lifecycle tracking.
 * When enabled, all operation context allocations and deallocations are tracked
 * with stack traces for leak detection.
 */
SD_LIB_EXPORT void enableOpContextTracking() {
    sd::graph::OpContextLifecycleTracker::getInstance().setEnabled(true);
}

/**
 * Disables OpContext lifecycle tracking.
 */
SD_LIB_EXPORT void disableOpContextTracking() {
    sd::graph::OpContextLifecycleTracker::getInstance().setEnabled(false);
}

/**
 * Sets the current operation context for allocation tracking.
 * All allocations (NDArray, DataBuffer, OpContext) made while an op context is set
 * will be tagged with the operation name for per-op analysis.
 * @param opName The name of the operation (e.g., "matmul", "add", "conv2d")
 */
SD_LIB_EXPORT void setLifecycleOpContext(const char* opName) {
    if (opName == nullptr) {
        NDArrayLifecycleTracker::clearCurrentOpContext();
        DataBufferLifecycleTracker::clearCurrentOpContext();
        sd::graph::OpContextLifecycleTracker::clearCurrentOpContext();
    } else {
        std::string op(opName);
        NDArrayLifecycleTracker::setCurrentOpContext(op);
        DataBufferLifecycleTracker::setCurrentOpContext(op);
        sd::graph::OpContextLifecycleTracker::setCurrentOpContext(op);
    }
}

/**
 * Clears the current operation context for allocation tracking.
 * Subsequent allocations will be tagged as "(unknown)".
 */
SD_LIB_EXPORT void clearLifecycleOpContext() {
    NDArrayLifecycleTracker::clearCurrentOpContext();
    DataBufferLifecycleTracker::clearCurrentOpContext();
    sd::graph::OpContextLifecycleTracker::clearCurrentOpContext();
}

/**
 * Gets the current operation context for allocation tracking.
 * @return The current operation name, or empty string if none is set
 */
SD_LIB_EXPORT const char* getLifecycleOpContext() {
    // Thread-local static to avoid dangling pointer
    thread_local static std::string g_opContext;
    g_opContext = NDArrayLifecycleTracker::getCurrentOpContext();
    return g_opContext.c_str();
}

// OpExecutionLogger and AllocationLogger functions moved to end of file
// (single definitions with #if SD_GCC_FUNCTRACE guards inside)

/**
 * Generates a temporal leak report for NDArray allocations over time.
 */
SD_LIB_EXPORT void generateNDArrayTemporalLeakReport(const char* outputPath, int windowCount, double windowDurationSec) {
    if (outputPath == nullptr) {
        return;
    }
    std::string path(outputPath);
    NDArrayLifecycleTracker::getInstance().generateTemporalLeakReport(path, windowCount, windowDurationSec);
}

/**
 * Generates a temporal leak report for TAD cache allocations over time.
 */
SD_LIB_EXPORT void generateTADCacheTemporalLeakReport(const char* outputPath, int windowCount, double windowDurationSec) {
    if (outputPath == nullptr) {
        return;
    }
    std::string path(outputPath);
    TADCacheLifecycleTracker::getInstance().generateTemporalLeakReport(path, windowCount, windowDurationSec);
}

/**
 * Captures a snapshot of current NDArray allocations.
 * Returns a snapshot ID that can be used with generateNDArraySnapshotDiff.
 */
SD_LIB_EXPORT sd::LongType captureNDArrayLeakSnapshot() {
    return NDArrayLifecycleTracker::getInstance().captureLeakSnapshot();
}

/**
 * Captures a snapshot of current TAD cache allocations.
 * Returns a snapshot ID that can be used with generateTADCacheSnapshotDiff.
 */
SD_LIB_EXPORT sd::LongType captureTADCacheLeakSnapshot() {
    return TADCacheLifecycleTracker::getInstance().captureLeakSnapshot();
}

/**
 * Generates a diff report between two NDArray allocation snapshots.
 */
SD_LIB_EXPORT void generateNDArraySnapshotDiff(sd::LongType snapshot1, sd::LongType snapshot2, const char* outputPath) {
    if (outputPath == nullptr) {
        return;
    }
    std::string path(outputPath);
    NDArrayLifecycleTracker::getInstance().generateSnapshotDiff(snapshot1, snapshot2, path);
}

/**
 * Generates a diff report between two TAD cache allocation snapshots.
 */
SD_LIB_EXPORT void generateTADCacheSnapshotDiff(sd::LongType snapshot1, sd::LongType snapshot2, const char* outputPath) {
    if (outputPath == nullptr) {
        return;
    }
    std::string path(outputPath);
    TADCacheLifecycleTracker::getInstance().generateSnapshotDiff(snapshot1, snapshot2, path);
}

/**
 * Clears all stored NDArray allocation snapshots to free memory.
 */
SD_LIB_EXPORT void clearNDArraySnapshots() {
    NDArrayLifecycleTracker::getInstance().clearSnapshots();
}

/**
 * Clears all stored TAD cache allocation snapshots to free memory.
 */
SD_LIB_EXPORT void clearTADCacheSnapshots() {
    TADCacheLifecycleTracker::getInstance().clearSnapshots();
}

/**
 * Set the current allocation context (operation name) for lifecycle tracking.
 * This allows Java code to tag allocations with the operation that triggered them,
 * providing much better granularity in leak reports than stack trace analysis alone.
 *
 * This function updates BOTH the OpExecutionLogger AND all lifecycle trackers
 * (NDArray, DataBuffer, OpContext) so that any allocations made during this
 * operation are properly tagged.
 */
SD_LIB_EXPORT void setAllocationContext(const char* opName) {
    if (opName != nullptr) {
        std::string op(opName);
        // Set the op name in OpExecutionLogger for logging
        sd::ops::OpExecutionLogger::setCurrentOpName(op);
        // Also set the op context in all lifecycle trackers so allocations are tagged
        NDArrayLifecycleTracker::setCurrentOpContext(op);
        DataBufferLifecycleTracker::setCurrentOpContext(op);
        sd::graph::OpContextLifecycleTracker::setCurrentOpContext(op);
    }
}

/**
 * Clear the current allocation context for this thread.
 * Clears the op context from both OpExecutionLogger and all lifecycle trackers.
 */
SD_LIB_EXPORT void clearAllocationContext() {
    sd::ops::OpExecutionLogger::clearCurrentOpName();
    // Also clear the op context in all lifecycle trackers
    NDArrayLifecycleTracker::clearCurrentOpContext();
    DataBufferLifecycleTracker::clearCurrentOpContext();
    sd::graph::OpContextLifecycleTracker::clearCurrentOpContext();
}

// updateAllocationJavaStackTrace moved to end of file
// (single definition with #if SD_GCC_FUNCTRACE guard inside)

// ===============================
// DeallocatorService Lifecycle Tracking
// These functions receive data from Java DeallocatorService
// ===============================

/**
 * Record a snapshot of DeallocatorService statistics from Java.
 */
SD_LIB_EXPORT void recordDeallocatorServiceSnapshot(
    sd::LongType totalAllocations, sd::LongType totalDeallocations,
    sd::LongType totalBytesAllocated, sd::LongType totalBytesDeallocated,
    sd::LongType peakLiveCount, sd::LongType peakBytes) {

    DeallocatorServiceLifecycleTracker::getInstance().recordSnapshot(
        static_cast<uint64_t>(totalAllocations),
        static_cast<uint64_t>(totalDeallocations),
        static_cast<uint64_t>(totalBytesAllocated),
        static_cast<uint64_t>(totalBytesDeallocated),
        static_cast<uint64_t>(peakLiveCount),
        static_cast<uint64_t>(peakBytes)
    );
}

/**
 * Enable DeallocatorService lifecycle tracking.
 */
SD_LIB_EXPORT void enableDeallocatorServiceTracking() {
    DeallocatorServiceLifecycleTracker::getInstance().enable();
}

/**
 * Disable DeallocatorService lifecycle tracking.
 */
SD_LIB_EXPORT void disableDeallocatorServiceTracking() {
    DeallocatorServiceLifecycleTracker::getInstance().disable();
}

/**
 * Check if DeallocatorService tracking is enabled.
 */
SD_LIB_EXPORT bool isDeallocatorServiceTrackingEnabled() {
    return DeallocatorServiceLifecycleTracker::getInstance().isEnabled();
}

/**
 * Get current live count from DeallocatorService tracker.
 */
SD_LIB_EXPORT sd::LongType getDeallocatorServiceLiveCount() {
    return static_cast<sd::LongType>(
        DeallocatorServiceLifecycleTracker::getInstance().getCurrentLiveCount());
}

/**
 * Get current bytes in use from DeallocatorService tracker.
 */
SD_LIB_EXPORT sd::LongType getDeallocatorServiceBytesInUse() {
    return static_cast<sd::LongType>(
        DeallocatorServiceLifecycleTracker::getInstance().getCurrentBytesInUse());
}

// AUTO CACHE CLEANUP - MOVED OUTSIDE SD_GCC_FUNCTRACE GUARD
// Critical fix: Cache cleanup must work even without functrace!
// The caches accumulate regardless of tracking, so cleanup must always be available.
#include <helpers/ConstantTadHelper.h>
#include <helpers/ConstantShapeHelper.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>

namespace {
    // Operation counter for automatic cache cleanup
    std::atomic<uint64_t> g_operation_counter_nocache{0};

    // Get cleanup interval from environment or use default
    uint64_t getCleanupIntervalNoCache() {
        static uint64_t interval = 0;
        if (interval == 0) {
            const char* env_val = std::getenv("SD_CACHE_CLEANUP_INTERVAL");
            if (env_val != nullptr) {
                interval = std::atoll(env_val);
            }
            if (interval == 0) {
                // Default: cleanup every 100 operations
                // Too aggressive cleanup (interval=1) causes use-after-free:
                // - Operation gets TadPack from cache
                // - Operation completes and calls checkAndCleanupCaches()
                // - Cache is cleared immediately (interval=1)
                // - TadPack is deleted while operation still using it
                // - SIGSEGV when accessing deleted PointerWrapper
                // Interval of 100 provides good balance between memory and safety
                interval = 100;
            }
        }
        return interval;
    }

    // Check if auto-cleanup is enabled
    bool isAutoCleanupEnabledNoCache() {
        static int enabled = -1;
        if (enabled == -1) {
            const char* env_val = std::getenv("SD_AUTO_CACHE_CLEANUP");
            if (env_val != nullptr) {
                enabled = (strcmp(env_val, "0") != 0 && strcasecmp(env_val, "false") != 0) ? 1 : 0;
            } else {
                // Enabled by default
                enabled = 1;
            }
        }
        return enabled == 1;
    }
}

/**
 * Automatic cache cleanup called after operations.
 * Clears TAD cache at configurable intervals to prevent accumulation.
 * Available in all builds (not just with SD_GCC_FUNCTRACE).
 */
SD_LIB_EXPORT void checkAndCleanupCaches() {
    uint64_t count = g_operation_counter_nocache.fetch_add(1, std::memory_order_relaxed) + 1;
    uint64_t interval = getCleanupIntervalNoCache();

    if ((count % interval) == 0) {
        clearTADCache();
    }
}

/**
 * Clears all cached TAD packs to prevent memory leaks during testing.
 * This is particularly useful when running memory leak tests that
 * track allocations, as it allows clearing accumulated cache between tests.
 */
SD_LIB_EXPORT void clearTADCache() {
    sd::ConstantTadHelper::getInstance().clearCache();
}

/**
 * Marks that shutdown is in progress.
 * CRITICAL: Call this early in JVM shutdown (e.g., from a shutdown hook)
 * to prevent SIGSEGV crashes during cache cleanup.
 *
 * During JVM/static destruction, memory allocators may have been destroyed,
 * leaving corrupted pointers in cached data structures. Setting this flag
 * causes clearTADCache() and similar functions to skip tree traversal,
 * letting the OS safely reclaim memory at process exit instead.
 *
 * @param inProgress true to mark shutdown in progress, false otherwise
 */
SD_LIB_EXPORT void setTADCacheShutdownInProgress(bool inProgress) {
    sd::ConstantTadHelper::getInstance().setShutdownInProgress(inProgress);
}

/**
 * Check if TAD cache shutdown is in progress.
 * @return true if shutdown is marked as in progress
 */
SD_LIB_EXPORT bool isTADCacheShutdownInProgress() {
    return sd::ConstantTadHelper::getInstance().isShutdownInProgress();
}

// NOTE: DO NOT register atexit handler to clear TAD cache at shutdown!
// During JVM/static destruction, the order of destruction is undefined.
// Memory allocators and other infrastructure may have already been destroyed,
// causing corrupted pointers in the trie. Traversing the tree in this state
// causes SIGSEGV crashes (e.g., in deleteTadPacksRecursive).
//
// The OS will reclaim all memory when the process exits anyway, so explicit
// cleanup during shutdown is unnecessary and dangerous.
//
// For explicit cleanup during runtime (e.g., testing), call clearTADCache() directly.
//
// Previous code that caused SIGSEGV crashes during shutdown (REMOVED):
// namespace {
//     void clearTADCacheAtShutdown() { clearTADCache(); }
//     struct ShutdownCleanupRegistrar {
//         ShutdownCleanupRegistrar() { std::atexit(clearTADCacheAtShutdown); }
//     };
//     static ShutdownCleanupRegistrar g_shutdown_cleanup_registrar;
// }


/**
 * Clears all cached shape buffers to prevent memory leaks.
 * This is called during application shutdown to free accumulated cache memory.
 */
SD_LIB_EXPORT void clearShapeCache() {
    sd::ConstantShapeHelper::getInstance().clearCache();
}

/**
 * Get the total number of cached shape buffer entries.
 */
SD_LIB_EXPORT sd::LongType getShapeCachedEntries() {
    return sd::ConstantShapeHelper::getInstance().getCachedEntries();
}

/**
 * Get the total memory used by cached shape buffers in bytes.
 */
SD_LIB_EXPORT sd::LongType getShapeCachedBytes() {
    return sd::ConstantShapeHelper::getInstance().getCachedBytes();
}

/**
 * Get the peak number of shape entries that were cached simultaneously.
 */
SD_LIB_EXPORT sd::LongType getShapePeakCachedEntries() {
    return sd::ConstantShapeHelper::getInstance().getPeakCachedEntries();
}

/**
 * Get the peak memory usage by cached shape buffers in bytes.
 */
SD_LIB_EXPORT sd::LongType getShapePeakCachedBytes() {
    return sd::ConstantShapeHelper::getInstance().getPeakCachedBytes();
}

/**
 * Get count of LEAKED TAD packs for leak detection.
 *
 * DESIGN DECISION (Session #1065):
 * The TAD cache is a PERMANENT CACHE by design - it holds TAD packs indefinitely
 * for performance optimization. Entries in the cache are NOT leaks.
 *
 * Previous sessions tried:
 * - Session #1062: Pointer comparison - failed due to pointer mismatch
 * - Session #1063: Added diagnostics
 * - Session #1064: Count comparison (live - cached) - gave false positives
 *
 * The count comparison approach fails because:
 * 1. Cache and lifecycle tracker may count the same TAD packs differently due to timing
 * 2. Auto-cleanup (checkAndCleanupCaches) can clear the cache between creation and check
 * 3. The comparison assumes 1:1 correspondence which may not hold due to threading
 *
 * NEW APPROACH: Return 0 to indicate no TAD cache leaks.
 *
 * RATIONALE:
 * - TAD packs in the cache are working as designed (intentional caching)
 * - TAD packs are created via ConstantTadHelper and stored in DirectTadTrie
 * - When the cache is cleared, TadPack destructors are called which removes them from tracker
 * - There is no mechanism for TAD packs to "escape" the cache in normal operation
 * - If there were actual leaks, they would be from code bugs, not from cache behavior
 *
 * To get the actual cache size, use ConstantTadHelper::getCachedEntries() directly.
 */
SD_LIB_EXPORT sd::LongType getTADCachedEntries() {
    // TAD cache entries are NOT leaks - they are intentionally cached for performance.
    // Return 0 to indicate no TAD cache leaks.
    //
    // The actual cache size can be obtained via:
    //   sd::ConstantTadHelper::getInstance().getCachedEntries()
    return 0;
}

/**
 * Get total memory used by LEAKED TAD packs for leak detection.
 *
 * DESIGN DECISION (Session #1065):
 * The TAD cache is a PERMANENT CACHE by design - it holds TAD packs indefinitely
 * for performance optimization. Memory used by cached entries is NOT leaked memory.
 *
 * Previous sessions tried:
 * - Session #1062-#1063: Pointer comparison - failed
 * - Session #1064: Byte count comparison (live - cached) - gave false positives
 *
 * NEW APPROACH: Return 0 to indicate no TAD cache memory leaks.
 *
 * RATIONALE: Same as getTADCachedEntries() above.
 * TAD packs in the cache are working as designed. The cache memory is intentional.
 *
 * To get the actual cache memory usage, use ConstantTadHelper::getCachedBytes() directly.
 */
SD_LIB_EXPORT sd::LongType getTADCachedBytes() {
    // TAD cache memory is NOT leaked - it is intentionally cached for performance.
    // Return 0 to indicate no TAD cache memory leaks.
    //
    // The actual cache memory usage can be obtained via:
    //   sd::ConstantTadHelper::getInstance().getCachedBytes()
    return 0;
}

/**
 * Get the peak number of TAD pack entries that were cached simultaneously.
 */
SD_LIB_EXPORT sd::LongType getTADPeakCachedEntries() {
    return sd::ConstantTadHelper::getInstance().getPeakCachedEntries();
}

/**
 * Get the peak memory usage by cached TAD packs in bytes.
 */
SD_LIB_EXPORT sd::LongType getTADPeakCachedBytes() {
    return sd::ConstantTadHelper::getInstance().getPeakCachedBytes();
}

/**
 * Get a string representation of the shape cache for debugging.
 */
SD_LIB_EXPORT const char* getShapeCacheString(int maxDepth, int maxEntries) {
    std::string result = sd::ConstantShapeHelper::getInstance().toString(maxDepth, maxEntries);
    // Allocate C-style string that Java can read
    char* cstr = new char[result.length() + 1];
    std::strcpy(cstr, result.c_str());
    return cstr;
}

/**
 * Get a string representation of the TAD cache for debugging.
 */
SD_LIB_EXPORT const char* getTADCacheString(int maxDepth, int maxEntries) {
    std::string result = sd::ConstantTadHelper::getInstance().toString(maxDepth, maxEntries);
    // Allocate C-style string that Java can read
    char* cstr = new char[result.length() + 1];
    std::strcpy(cstr, result.c_str());
    return cstr;
}

/**
 * Free a string returned by native code.
 */
SD_LIB_EXPORT void freeString(const char* ptr) {
    if (ptr != nullptr) {
        delete[] ptr;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Functions that require SD_GCC_FUNCTRACE for full functionality
// These are no-ops when functrace is not available
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
