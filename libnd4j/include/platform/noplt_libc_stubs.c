/*
 * noplt_libc_stubs.c - No-PLT stub functions for libc functions
 *
 * When building with SD_GCC_FUNCTRACE enabled, the binary grows to ~3GB which
 * exceeds the 2GB limit for PLT (Procedure Linkage Table) relocations on x86_64.
 *
 * The standard libc_nonshared.a library contains precompiled implementations of
 * functions like atexit() that use PLT relocations. When linked into a binary
 * larger than 2GB, these PLT-based relocations fail with:
 *   "relocation truncated to fit: R_X86_64_PLT32"
 *
 * This file provides stub implementations that will be compiled without PLT
 * (using -mcmodel=medium which implies GOT-based calls instead of PLT).
 * These symbols take precedence over the libc_nonshared.a versions due to
 * standard linker symbol resolution order (user code before libraries).
 *
 * The stubs use the __cxa_atexit interface directly, which is provided by
 * libc.so and doesn't have the PLT relocation issues of libc_nonshared.a.
 *
 * References:
 *   - https://itanium-cxx-abi.github.io/cxx-abi/abi.html#dso-dtor (__cxa_atexit)
 *   - https://refspecs.linuxbase.org/LSB_3.1.0/LSB-Core-generic/LSB-Core-generic/baselib---cxa-atexit.html
 */

/*
 * __cxa_atexit - Low-level C++ ABI function for registering destructors
 *
 * This function is part of the Itanium C++ ABI and is used by the compiler
 * to register static object destructors. It's provided by libc.so (not
 * libc_nonshared.a) and therefore doesn't have PLT relocation issues.
 *
 * Parameters:
 *   func - Destructor function to call (takes object and dso_handle)
 *   obj  - Object to pass to destructor
 *   dso  - DSO handle for the shared library
 *
 * Returns:
 *   0 on success, non-zero on failure
 */
extern int __cxa_atexit(void (*func)(void *), void *obj, void *dso);

/*
 * __dso_handle - DSO handle for the current shared object
 *
 * This symbol is provided by the compiler/linker and identifies the
 * current shared library for __cxa_atexit registration.
 */
extern void *__dso_handle;

/*
 * Adapter function for atexit callbacks
 *
 * __cxa_atexit expects a function that takes a void* argument,
 * but atexit functions take no arguments. This adapter ignores
 * the argument and calls the original function.
 */
typedef void (*atexit_func_t)(void);

static void atexit_adapter(void *func_ptr) {
    atexit_func_t func = (atexit_func_t)func_ptr;
    if (func) {
        func();
    }
}

/*
 * atexit - Register a function to be called at normal program termination
 *
 * This implementation uses __cxa_atexit internally, which avoids the PLT
 * relocation issues present in libc_nonshared.a's implementation.
 *
 * Parameters:
 *   function - Function to call at normal program termination
 *
 * Returns:
 *   0 on success, non-zero on failure
 */
int atexit(void (*function)(void)) {
    /*
     * Use __cxa_atexit with our DSO handle.
     * The adapter function will be called with the original function pointer,
     * which it then calls with no arguments.
     */
    return __cxa_atexit(atexit_adapter, (void *)function, __dso_handle);
}

/*
 * at_quick_exit - Register a function to be called at quick_exit()
 *
 * For functrace builds, we provide a minimal implementation that just
 * registers the function with the standard atexit mechanism.
 * This is acceptable because:
 * 1. quick_exit() is rarely used in practice
 * 2. The main purpose is to avoid PLT relocations, not provide perfect semantics
 *
 * Parameters:
 *   function - Function to call at quick_exit()
 *
 * Returns:
 *   0 on success, non-zero on failure
 */
int at_quick_exit(void (*function)(void)) {
    /*
     * Simplification: Register with regular atexit.
     * In a complete implementation, these would be separate lists,
     * but for functrace debugging purposes this is sufficient.
     */
    return atexit(function);
}
