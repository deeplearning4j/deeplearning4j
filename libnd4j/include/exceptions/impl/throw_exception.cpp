//
// Created by agibsonccc on 5/11/23.
//
#include <system/op_boilerplate.h>
#include <execution/LaunchContext.h>
#include <exceptions/backward.hpp>

using namespace backward;

// Helper function to safely check if LaunchContext is initialized
// Returns true if it's safe to use LaunchContext, false otherwise
static bool isLaunchContextReady() {
  // During early initialization (e.g., static initializers, before main()),
  // LaunchContext may not be initialized yet. Attempting to use it can cause
  // crashes. This function provides a safe check.

  // DON'T try to call LaunchContext::defaultContext() here!
  // Even calling it can trigger crashes during early initialization.
  // The safer approach is to just return false during exception handling
  // and let fprintf write to stderr instead.

  // We can't safely determine if LaunchContext is ready without potentially
  // triggering the same initialization issues we're trying to avoid.
  // So we conservatively return false and use fprintf for error reporting.
  return false;
}

// Safe helper function for setting error context in exception handlers
// This should be used in ALL catch blocks in JNI code instead of directly
// calling LaunchContext::defaultContext()->errorReference()
void safeSetErrorContext(int errorCode, const char* errorMessage) {
  if (isLaunchContextReady()) {
#ifdef __cpp_exceptions
    try {
      sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(errorCode);
      sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(errorMessage);
    } catch (...) {
      // If setting error context fails, just log to stderr
      // Don't let this cause another exception
      fprintf(stderr, "Warning: Failed to set error context (code=%d): %s\n", errorCode, errorMessage);
    }
#else
    // Exceptions disabled - direct call without try/catch
    sd::LaunchContext::defaultContext()->errorReference()->setErrorCode(errorCode);
    sd::LaunchContext::defaultContext()->errorReference()->setErrorMessage(errorMessage);
#endif
  } else {
    // LaunchContext not ready - just log to stderr
    // This is normal during early JVM initialization
    fprintf(stderr, "Error (code=%d) during early initialization: %s\n", errorCode, errorMessage);
  }
}

#if defined(SD_GCC_FUNCTRACE)
void throwException(const char* exceptionMessage) {
#ifndef __CUDA_CC__
  // Diagnostic: verify this code path executes
  fprintf(stderr, "\n[throwException] SD_GCC_FUNCTRACE build - capturing stack trace\n");
  fflush(stderr);

  // Print exception message first
  fprintf(stderr, "=== EXCEPTION: %s ===\n", exceptionMessage);
  fflush(stderr);

  // Capture and print stack trace
  StackTrace st;
  st.load_here(64);

  if (st.size() > 0) {
    fprintf(stderr, "Stack trace (%zu frames):\n", st.size());
    fflush(stderr);
    Printer p;
    p.snippet = false;
    p.color_mode = ColorMode::never;
    p.address = true;
    p.object = true;
    p.print(st, stderr);
    fflush(stderr);
  } else {
    fprintf(stderr, "Stack trace: Unable to capture (0 frames captured)\n");
    fprintf(stderr, "This may indicate missing unwind tables or debug info\n");
    fflush(stderr);
  }
  fprintf(stderr, "=== END STACK TRACE ===\n\n");
  fflush(stderr);
#endif

  // Set error context for Java to retrieve using safe wrapper
  // CRITICAL: Don't directly call LaunchContext during exception handling
  // to avoid static initialization/destruction order issues
  safeSetErrorContext(1, exceptionMessage);

#ifdef __cpp_exceptions
  throw std::runtime_error(exceptionMessage);
#endif
}
#else
void throwException(const char* exceptionMessage) {
  // Diagnostic: verify this code path executes
  fprintf(stderr, "\n[throwException] Non-functrace build - no stack trace available\n");
  fprintf(stderr, "Exception: %s\n", exceptionMessage);
  fflush(stderr);

  // Set error context for Java to retrieve using safe wrapper
  // CRITICAL: Don't directly call LaunchContext during exception handling
  // to avoid static initialization/destruction order issues
  safeSetErrorContext(1, exceptionMessage);

#ifdef __cpp_exceptions
  throw std::runtime_error(exceptionMessage);
#endif
}
#endif