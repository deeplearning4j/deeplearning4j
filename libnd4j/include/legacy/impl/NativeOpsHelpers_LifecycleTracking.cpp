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

#if defined(SD_GCC_FUNCTRACE)

// Forward declare the ComprehensiveLeakAnalyzer class before including tracker headers
// This ensures the friend declarations in the tracker classes can see the class
namespace sd {
namespace analysis {
    class ComprehensiveLeakAnalyzer;
}
}

#include <array/NDArrayLifecycleTracker.h>
#include <array/DataBufferLifecycleTracker.h>
#include <array/TADCacheLifecycleTracker.h>
#include <array/ShapeCacheLifecycleTracker.h>
#include <graph/OpContextLifecycleTracker.h>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <atomic>
#include <cstdlib>
#include <thread>
#include <csignal>
#include <unistd.h>
#include <fcntl.h>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <fstream>
#include <array>
#include <vector>
#include <sched.h>
#include <iostream>
#ifndef _WIN32
#include <pthread.h>
#endif
#ifdef __linux__
#include <sys/syscall.h>
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
        sa.sa_flags = SA_SIGINFO | SA_ONSTACK;

        for (size_t i = 0; i < kSignalCount; ++i) {
            if (sigaction(kCrashSignals[i], &sa, &_oldHandlers[i]) != 0) {
                std::cerr << "[sd-crash] Failed to install handler for signal "
                          << kCrashSignals[i] << std::endl;
            }
        }
        _handlersInstalled = true;
    }

    static void signalHandler(int signo, siginfo_t* info, void* /*ucontext*/) {
        LifecycleCrashHandler::instance().handleSignal(signo, info);
    }

    void handleSignal(int signo, siginfo_t* info) {
        if (!_ready.load(std::memory_order_acquire)) {
            restoreAndReraise(signo);
            return;
        }

        if (_handling.exchange(true, std::memory_order_acq_rel)) {
            restoreAndReraise(signo);
            return;
        }

        CrashEvent event{};
        event.signal = signo;
        event.faultAddress = info ? info->si_addr : nullptr;
        event.crashingThreadId = currentThreadId();

        _dumpComplete.store(false, std::memory_order_release);

        ssize_t wrote = ::write(_signalPipe[1], &event, sizeof(event));
        if (wrote != sizeof(event)) {
            _handling.store(false, std::memory_order_release);
            restoreAndReraise(signo);
            return;
        }

        while (!_dumpComplete.load(std::memory_order_acquire)) {
            sched_yield();
        }

        _handling.store(false, std::memory_order_release);
        restoreAndReraise(signo);
    }

    void restoreAndReraise(int signo) {
        restoreOriginal(signo);
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
#endif

struct CrashHandlerAutoInit {
    CrashHandlerAutoInit() {
        LifecycleCrashHandler::instance().ensureInitialized();
    }
};

static CrashHandlerAutoInit gCrashHandlerAutoInit;

}  // namespace
#endif

// AUTO CACHE CLEANUP: Moved outside #if defined(SD_GCC_FUNCTRACE) guard
// NOTE: The duplicate namespace block with g_operation_counter and helper functions
// has been removed. The implementation now uses the "_nocache" versions defined
// outside the SD_GCC_FUNCTRACE guard (see around line 424) to ensure cleanup works
// in all builds, not just functrace builds.

// NOTE: checkAndCleanupCaches() has been moved OUTSIDE the #if defined(SD_GCC_FUNCTRACE) guard
// to ensure it's available in all builds (see implementation around line 478).
// Previous duplicate implementation here has been removed to fix linker symbol conflict.
//
// Forward declarations for cache clearing functions (implementations are outside this guard)
SD_LIB_EXPORT void clearTADCache();
SD_LIB_EXPORT void clearShapeCache();
SD_LIB_EXPORT void checkAndCleanupCaches();

// Include comprehensive leak analysis implementation
#include "../generate_leak_analysis.cpp"

/**
 * Converts NDArray lifecycle statistics to JSON format.
 */
const char* getNDArrayLifecycleStats() {
    auto stats = NDArrayLifecycleTracker::getInstance().getStats();

    std::ostringstream json;
    json << "{\n";
    json << "  \"total_allocations\": " << stats.total_allocs << ",\n";
    json << "  \"total_deallocations\": " << stats.total_deallocs << ",\n";
    json << "  \"current_live\": " << stats.current_live << ",\n";
    json << "  \"current_bytes\": " << stats.current_bytes << ",\n";
    json << "  \"peak_bytes\": " << stats.peak_bytes << ",\n";
    json << "  \"double_frees\": " << stats.double_frees << "\n";
    json << "}";

    std::string result = json.str();
    char* cstr = new char[result.length() + 1];
    std::strcpy(cstr, result.c_str());
    return cstr;
}

/**
 * Converts DataBuffer lifecycle statistics to JSON format.
 */
const char* getDataBufferLifecycleStats() {
    auto stats = DataBufferLifecycleTracker::getInstance().getStats();

    std::ostringstream json;
    json << "{\n";
    json << "  \"primary\": {\n";
    json << "    \"total_allocations\": " << stats.total_primary_allocs << ",\n";
    json << "    \"total_deallocations\": " << stats.total_primary_deallocs << ",\n";
    json << "    \"current_live\": " << stats.current_live_primary << ",\n";
    json << "    \"current_bytes\": " << stats.current_primary_bytes << ",\n";
    json << "    \"peak_bytes\": " << stats.peak_primary_bytes << "\n";
    json << "  },\n";
    json << "  \"special\": {\n";
    json << "    \"total_allocations\": " << stats.total_special_allocs << ",\n";
    json << "    \"total_deallocations\": " << stats.total_special_deallocs << ",\n";
    json << "    \"current_live\": " << stats.current_live_special << ",\n";
    json << "    \"current_bytes\": " << stats.current_special_bytes << ",\n";
    json << "    \"peak_bytes\": " << stats.peak_special_bytes << "\n";
    json << "  },\n";
    json << "  \"double_frees\": " << stats.double_frees << "\n";
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
    DataBufferLifecycleTracker::getInstance().generateFlamegraph(path, type);
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
    DataBufferLifecycleTracker::getInstance().generateDeletionFlamegraph(path, type);
}

/**
 * Generates a comprehensive leak report combining NDArray and DataBuffer leaks.
 */
void generateLifecycleLeakReport(const char* outputPath) {
    if (outputPath == nullptr) {
        return;
    }

    std::string path(outputPath);

    // Generate separate reports then combine them
    std::string ndarray_report = path + ".ndarray.txt";
    std::string databuffer_report = path + ".databuffer.txt";

    // Generate individual reports
    NDArrayLifecycleTracker::getInstance().generateLeakReport(ndarray_report);
    DataBufferLifecycleTracker::getInstance().generateLeakReport(databuffer_report);

    // Create combined report
    std::ofstream combined(path);
    if (combined.is_open()) {
        combined << "============================================\n";
        combined << "  COMBINED LIFECYCLE LEAK REPORT\n";
        combined << "============================================\n\n";

        // NDArray statistics
        auto ndarray_stats = NDArrayLifecycleTracker::getInstance().getStats();
        combined << "NDArray Statistics:\n";
        combined << "  Total Allocations:   " << ndarray_stats.total_allocs << "\n";
        combined << "  Total Deallocations: " << ndarray_stats.total_deallocs << "\n";
        combined << "  Current Live:        " << ndarray_stats.current_live << "\n";
        combined << "  Current Bytes:       " << ndarray_stats.current_bytes << "\n";
        combined << "  Peak Bytes:          " << ndarray_stats.peak_bytes << "\n";
        combined << "  Double Frees:        " << ndarray_stats.double_frees << "\n\n";

        // DataBuffer statistics
        auto databuffer_stats = DataBufferLifecycleTracker::getInstance().getStats();
        combined << "DataBuffer Statistics:\n";
        combined << "  PRIMARY (Host) Memory:\n";
        combined << "    Total Allocations:   " << databuffer_stats.total_primary_allocs << "\n";
        combined << "    Total Deallocations: " << databuffer_stats.total_primary_deallocs << "\n";
        combined << "    Current Live:        " << databuffer_stats.current_live_primary << "\n";
        combined << "    Current Bytes:       " << databuffer_stats.current_primary_bytes << "\n";
        combined << "    Peak Bytes:          " << databuffer_stats.peak_primary_bytes << "\n";
        combined << "  SPECIAL (Device) Memory:\n";
        combined << "    Total Allocations:   " << databuffer_stats.total_special_allocs << "\n";
        combined << "    Total Deallocations: " << databuffer_stats.total_special_deallocs << "\n";
        combined << "    Current Live:        " << databuffer_stats.current_live_special << "\n";
        combined << "    Current Bytes:       " << databuffer_stats.current_special_bytes << "\n";
        combined << "    Peak Bytes:          " << databuffer_stats.peak_special_bytes << "\n";
        combined << "  Double Frees:          " << databuffer_stats.double_frees << "\n\n";

        combined << "See detailed reports:\n";
        combined << "  " << ndarray_report << "\n";
        combined << "  " << databuffer_report << "\n";

        combined.close();
    }
}

/**
 * Generates a comprehensive leak source analysis combining ALL lifecycle trackers.
 */
void generateComprehensiveLeakAnalysis(const char* outputDir) {
    if (outputDir == nullptr) {
        return;
    }

    std::string dir(outputDir);
    runComprehensiveLeakAnalysis(dir.c_str());
}

#endif // SD_GCC_FUNCTRACE

// Enable/disable functions are ALWAYS available, regardless of SD_GCC_FUNCTRACE
// They will simply enable/disable tracking if the trackers are compiled in

#if defined(SD_GCC_FUNCTRACE)

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

#endif // SD_GCC_FUNCTRACE

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

// Forward declarations for cache clearing
SD_LIB_EXPORT void clearTADCache();
SD_LIB_EXPORT void clearShapeCache();

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

// Register atexit handler to clear TAD cache at shutdown
// This ensures cache is empty when application exits, preventing false leak reports
namespace {
    void clearTADCacheAtShutdown() {
        clearTADCache();
    }

    struct ShutdownCleanupRegistrar {
        ShutdownCleanupRegistrar() {
            std::atexit(clearTADCacheAtShutdown);
        }
    };

    static ShutdownCleanupRegistrar g_shutdown_cleanup_registrar;
}


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
 * Get count of live TAD packs for leak detection.
 * When tracking enabled: returns actual live count from lifecycle tracker
 * When tracking disabled: returns cache size as fallback
 */
SD_LIB_EXPORT sd::LongType getTADCachedEntries() {
#if defined(SD_GCC_FUNCTRACE) && !defined(__JAVACPP_HACK__)
    auto stats = sd::array::TADCacheLifecycleTracker::getInstance().getStats();
    return static_cast<sd::LongType>(stats.current_live);
#else
    return sd::ConstantTadHelper::getInstance().getCachedEntries();
#endif
}

/**
 * Get total memory used by live TAD packs for leak detection.
 * When tracking enabled: returns actual live bytes from lifecycle tracker
 * When tracking disabled: returns cache bytes as fallback
 */
SD_LIB_EXPORT sd::LongType getTADCachedBytes() {
#if defined(SD_GCC_FUNCTRACE) && !defined(__JAVACPP_HACK__)
    auto stats = sd::array::TADCacheLifecycleTracker::getInstance().getStats();
    return static_cast<sd::LongType>(stats.current_bytes);
#else
    return sd::ConstantTadHelper::getInstance().getCachedBytes();
#endif
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

#if !defined(SD_GCC_FUNCTRACE)
// When SD_GCC_FUNCTRACE is not defined - lifecycle tracking disabled

// When SD_GCC_FUNCTRACE is not defined, trackers don't exist, so these are no-ops
SD_LIB_EXPORT void enableNDArrayTracking() {}
SD_LIB_EXPORT void disableNDArrayTracking() {}
SD_LIB_EXPORT void enableDataBufferTracking() {}
SD_LIB_EXPORT void disableDataBufferTracking() {}
SD_LIB_EXPORT void enableTADCacheTracking() {}
SD_LIB_EXPORT void disableTADCacheTracking() {}
SD_LIB_EXPORT void enableShapeCacheTracking() {}
SD_LIB_EXPORT void disableShapeCacheTracking() {}
SD_LIB_EXPORT void enableOpContextTracking() {}
SD_LIB_EXPORT void disableOpContextTracking() {}

// Stub implementations for lifecycle query and generation functions
// These return empty/null values when tracking is not compiled in
// NOTE: checkAndCleanupCaches() is now always available (moved outside #ifdef block)
// to ensure TAD cache cleanup works in all builds, not just functrace builds.

SD_LIB_EXPORT const char* getNDArrayLifecycleStats() {
    // Return empty JSON object when tracking is disabled
    static const char* empty_stats = "{}";
    return empty_stats;
}

SD_LIB_EXPORT const char* getDataBufferLifecycleStats() {
    // Return empty JSON object when tracking is disabled
    static const char* empty_stats = "{}";
    return empty_stats;
}

SD_LIB_EXPORT void generateNDArrayAllocationFlamegraph(const char* outputPath) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void generateNDArrayDeallocationFlamegraph(const char* outputPath) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void generateDataBufferAllocationFlamegraph(const char* outputPath, int bufferType) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void generateDataBufferDeallocationFlamegraph(const char* outputPath, int bufferType) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void generateLifecycleLeakReport(const char* outputPath) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void generateComprehensiveLeakAnalysis(const char* outputDir) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void generateNDArrayTemporalLeakReport(const char* outputPath, int windowCount, double windowDurationSec) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void generateTADCacheTemporalLeakReport(const char* outputPath, int windowCount, double windowDurationSec) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT sd::LongType captureNDArrayLeakSnapshot() {
    return 0;
}

SD_LIB_EXPORT sd::LongType captureTADCacheLeakSnapshot() {
    return 0;
}

SD_LIB_EXPORT void generateNDArraySnapshotDiff(sd::LongType snapshot1, sd::LongType snapshot2, const char* outputPath) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void generateTADCacheSnapshotDiff(sd::LongType snapshot1, sd::LongType snapshot2, const char* outputPath) {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void clearNDArraySnapshots() {
    // No-op when tracking is disabled
}

SD_LIB_EXPORT void clearTADCacheSnapshots() {
    // No-op when tracking is disabled
}

#endif // SD_GCC_FUNCTRACE
