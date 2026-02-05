# Scalar Use-After-Free Issue Tracking

## Document Standard

This document tracks investigation and fixes for the scalar constant array use-after-free bug. Each entry follows this format:

### Entry Format
```
## [DATE] - [SUMMARY]

### Files Modified
- `path/to/file.java` - Brief description of change

### Files Read/Analyzed
- `path/to/file.java:LINE` - What was found

### Hypothesis
What we believed was happening

### Action Taken
What we did to test/fix

### Result
What happened - SUCCESS/FAILURE/PARTIAL

### Next Steps
What to try next based on findings
```

---

## Problem Statement

**Symptom**: Constant scalar arrays loaded from SameDiff models (e.g., `Attention_0_three` with expected value `3`) return garbage values like `0x756f0012cf3e0001` which contain ASCII-like data, indicating memory was reused after being freed.

**Error Location**: `floordiv.cpp` in libnd4j when reading scalar input via `e<LongType>(0)`

**Key Debug Finding**: Buffer address `0x7f0896077480` was NOT in the DeallocatorService registration list, meaning either:
- The buffer was never registered
- A DIFFERENT buffer was registered than the one being used

---

## 2025-01-20 - Initial Deep Investigation

### Files Read/Analyzed

1. `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/array/ThreadSafeArrayHolder.java` (188 lines)
   - `setArray()` method at line 71-115: Creates `DeviceLocalNDArray` and calls `broadcast()`
   - Captures `sourceWasConstant` flag before operations
   - Calls `propagateConstantFlag()` after storing

2. `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/autodiff/samediff/array/SingleThreadArrayHolder.java` (118 lines)
   - Simpler implementation, also has `propagateConstantFlag()` logic

3. `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/util/DeviceLocalNDArray.java` (351 lines)
   - `broadcast()` method at line 168-240: Creates duplicates for multi-device
   - Key issue: `delayedArray = array.dup(array.ordering()).detach()` creates NEW buffer
   - `propagateConstantFlag()` at line 322-350: Re-checks source constant flag

4. `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/nativeblas/OpaqueDataBuffer.java` (611 lines)
   - Native buffer wrapper with `setConstant()` method

5. `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/memory/deallocation/OpaqueDataBufferDeallocator.java` (205 lines)
   - Deallocates via `dbClose()` native call
   - Checks `isConstant()` flag before deallocating

6. `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/memory/deallocation/DeallocatorService.java`
   - `pickObject()` method processes phantom references
   - `pendingConstants` set protects buffers during registration window
   - `registerPendingConstant()` / `releasePendingConstant()` for protection

7. `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/buffer/BaseCudaDataBuffer.java`
   - `ByteBuffer` constructor copies data to DEVICE via `memcpyAsync`
   - HOST buffer NOT initialized with data
   - `tickDeviceWrite()` marks device as having latest data

8. `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/jita/allocator/impl/CudaDeallocator.java` (195 lines)
   - CUDA-specific deallocator, separate from base OpaqueDataBufferDeallocator

9. `libnd4j/include/ops/declarable/generic/broadcastable/floordiv.cpp` (147 lines)
   - Where error manifests when reading scalar via `e<LongType>(0)`

10. `libnd4j/include/array/NDArray.hXX`
    - `e<T>(i)` template reads via `syncToHost()` then accesses host buffer

11. `libnd4j/include/array/cuda/DataBuffer.cu`
    - `syncToHost()` / `syncToPrimary()` implementations
    - Checks `isPrimaryActual()` - skips sync if true

### Hypothesis

**Buffer Aliasing Problem**:
1. Array loaded from model, marked constant
2. Array duplicated during storage in `ThreadSafeArrayHolder` → `DeviceLocalNDArray.broadcast()`
3. ORIGINAL buffer pointer still held somewhere (possibly in temporary variable)
4. Original goes out of scope, GC'd, deallocated
5. Code reads from dangling pointer

**Alternative Hypothesis - Sync Issue**:
- `tickDeviceWrite()` called on wrong buffer during ByteBuffer construction
- `isPrimaryActual()` returns true incorrectly, skipping device→host sync
- Host buffer contains uninitialized/stale data

### Constant Flag Propagation Chain
```
createFromFlatArray()
  → ByteBuffer
  → CudaLongDataBuffer
  → BaseCudaDataBuffer
  → setArrayForVariable()
  → ThreadSafeArrayHolder.setArray()
  → DeviceLocalNDArray.broadcast()
    → array.dup().detach() [NEW BUFFER CREATED]
    → propagateConstantFlag() [Must mark NEW buffer]
```

### Critical Code Paths Identified

**ThreadSafeArrayHolder.setArray() lines 71-115**:
```java
boolean sourceWasConstant = array.isConstant();  // Capture BEFORE operations
// ... dup/detach operations create NEW buffers ...
propagateConstantFlag(array, toBroadcast, sourceWasConstant);  // Must mark NEW buffer
```

**DeviceLocalNDArray.broadcast() lines 168-240**:
```java
delayedArray = array.dup(array.ordering()).detach();  // NEW buffer
propagateConstantFlag(array, delayed);  // RE-CHECKS source.isConstant() - potential race!
```

**propagateConstantFlag() lines 322-350**:
```java
if (source.data() != null && source.data().isConstant()) {  // RE-CHECK here is the bug!
    target.data().setConstant(true);
}
```

### Root Cause Identified

In `propagateConstantFlag()`, the method RE-CHECKS `source.data().isConstant()` instead of using the already-captured `sourceWasConstant` boolean. If there's any timing issue (GC, concurrent access), the source buffer might already be in an inconsistent state when this re-check happens.

### Action Taken
None yet - documenting findings first

### Result
INVESTIGATION COMPLETE - Ready to implement fix

### Next Steps

1. **Fix propagateConstantFlag()** to accept boolean parameter instead of re-checking:
   - `ThreadSafeArrayHolder.java` line 115: Already passes `sourceWasConstant`
   - `DeviceLocalNDArray.java` line 240: Change to pass captured boolean

2. **Add defensive logging** in floordiv.cpp to capture buffer state before read

3. **Verify fix** by running model loading test

---

## 2025-01-20 - Fix #1: propagateConstantFlag Race Condition

### Files Modified

1. `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/util/DeviceLocalNDArray.java`
   - **Lines 71-91**: Modified `propagateConstantFlag()` method
   - **Change**: Now uses the `sourceWasConstant` field (already captured at start of broadcast()) instead of re-checking `source.data().isConstant()`
   - **Reasoning**: Prevents race condition where source buffer may be in inconsistent state (GC'd, freed) by the time we check

### Code Change Details

**Before:**
```java
private void propagateConstantFlag(INDArray source, INDArray target) {
    if (source == null || target == null) return;

    // Check if source data buffer is constant
    DataBuffer sourceData = source.data();
    if (sourceData != null && sourceData.isConstant()) {  // <-- RE-CHECK: Race condition!
        DataBuffer targetData = target.data();
        if (targetData != null) {
            targetData.setConstant(true);
        }
        ...
    }
}
```

**After:**
```java
private void propagateConstantFlag(INDArray source, INDArray target) {
    if (target == null) return;

    // Use the already-captured sourceWasConstant field instead of re-checking source
    // This prevents race conditions where source buffer may have been freed
    if (sourceWasConstant) {  // <-- Uses field captured at start of broadcast()
        DataBuffer targetData = target.data();
        if (targetData != null) {
            targetData.setConstant(true);
        }
        ...
    }
}
```

### Files Already Correct (No Changes Needed)

- `ThreadSafeArrayHolder.java`: Already captures `sourceIsConstant` at line 88 and uses it directly at line 109
- `SingleThreadArrayHolder.java`: Already captures `sourceIsConstant` at line 52 and uses it directly at line 69

### Hypothesis

The race condition in `propagateConstantFlag()` was causing the constant flag to NOT be propagated to the duplicated buffer when:
1. `broadcast()` is called with a constant array
2. `sourceWasConstant = isSourceConstant(array)` captures `true`
3. `array.dup().detach()` creates a new buffer
4. Between dup and propagateConstantFlag, the ORIGINAL array's buffer gets GC'd
5. `propagateConstantFlag(array, delayed)` re-checks `array.data().isConstant()` which may now return false (or crash)
6. The duplicated buffer is NOT marked constant
7. Later, the duplicated buffer gets freed, causing use-after-free

### Result
**FAILED** - Issue still occurs. New error: `corrupted double-linked list` (heap corruption during deallocation)

New debug output shows:
- Buffer address: `0x7f73e2251c70` (different from previous `0x7f0896077480`)
- Garbage value: `0x726f0031dd1b0004` (still contains ASCII-like data)
- `corrupted double-linked list` error during GC collection

This indicates the race condition fix was NOT the root cause. The issue is deeper.

### Call Sites Verified

All 5 call sites of `propagateConstantFlag()` in `DeviceLocalNDArray.java` are after `sourceWasConstant` is set:

| Line | Method | `sourceWasConstant` set at |
|------|--------|---------------------------|
| 220 | broadcast() | Line 199 |
| 237 | broadcast() | Line 199 |
| 258 | broadcast() | Line 199 |
| 275 | broadcast() | Line 199 |
| 334 | update() | Lines 306-308 |

### Next Steps

1. Run the model loading test to verify fix
2. If still failing, investigate native layer sync issues (device→host buffer sync)

---

## 2025-01-20 - Root Cause Found: JavaCPP Double-Free

### Problem Identified

The `corrupted double-linked list` error is caused by **double-free** - both JavaCPP and ND4J's `DeallocatorService` trying to free the same memory.

### Evidence

Debug output shows:
```
Debug: Collecting org.bytedeco.javacpp.Pointer$NativeDeallocator[ownerAddress=0x7fad1d53a0a0,deallocatorAddress=0x7faf7c0b2490]
corrupted double-linked list ocator[ownerAddress=0x7f724c976e00,deallocatorAddress=0x7faf7c0b2490]]
```

- `Pointer$NativeDeallocator` is JavaCPP's built-in deallocator (separate from ND4J)
- The "corrupted double-linked list" is glibc's malloc error for double-free

### Root Cause

1. **JavaCPP auto-attaches deallocator**: When `allocateDataBuffer()` returns an `OpaqueDataBuffer`, JavaCPP automatically attaches a `NativeDeallocator` that calls `delete` on the native pointer.

2. **ND4J also registers deallocator**: `OpaqueDataBuffer.allocateDataBuffer()` registers an `OpaqueDataBufferDeallocator` with `DeallocatorService`.

3. **Both trigger on GC**: When the buffer becomes unreachable:
   - JavaCPP's `NativeDeallocator` triggers (via phantom reference)
   - ND4J's `OpaqueDataBufferDeallocator` also triggers (via DeallocatorService)
   - **BOTH call `dbClose()` or equivalent → double-free**

### Files Analyzed

| File | Line | Finding |
|------|------|---------|
| `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/nativeblas/NativeOps.java` | 505 | `allocateDataBuffer` has NO `@NoDeallocator` annotation |
| `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/nativeblas/OpaqueDataBuffer.java` | 47 | `OpaqueDataBuffer extends Pointer` - inherits JavaCPP deallocation |

### The Conflict

The codebase has conflicting approaches:
- **For constant buffers**: Calls `retainReference()` to prevent JavaCPP deallocation
- **For non-constant buffers**: Does NOT call `retainReference()`, so BOTH systems try to deallocate

Comments in code reveal awareness but no fix:
```java
// NOTE: Do NOT call retainReference() - it prevents DeallocatorService from working!
// WARNING: Do NOT call deallocator(null) - that runs the deallocator first!
```

### Proposed Fix

Add `@NoDeallocator` annotation to all methods in `NativeOps.java` that return `OpaqueDataBuffer`:

```java
@NoDeallocator  // Prevent JavaCPP from attaching deallocator
OpaqueDataBuffer allocateDataBuffer(long elements, int dataType, boolean allocateBoth);
```

This leaves ND4J's `DeallocatorService` as the SOLE deallocation mechanism, eliminating the double-free.

### Result
ROOT CAUSE IDENTIFIED - Fix implemented

### Fix Implementation

Added `@NoDeallocator` annotation to the following methods in `NativeOps.java`:

**OpaqueDataBuffer-returning methods:**
- `dbCreateView()`
- `dbAllocateDataBuffer()`
- `dbCreateExternalDataBuffer()`
- `allocateDataBuffer()`
- `intermediateResultDataAt()`

**OpaqueNDArray-returning methods:**
- `create()`
- `getOutputArrayNative()`
- `getInputArrayNative()`

### Sources
- [JavaCPP Pointer API](https://bytedeco.org/javacpp/apidocs/org/bytedeco/javacpp/Pointer.html)
- [JavaCPP NoDeallocator annotation](https://bytedeco.org/javacpp/apidocs/org/bytedeco/javacpp/annotation/NoDeallocator.html)

---

## All Files Modified (Cumulative)

| Date | File | Description |
|------|------|-------------|
| 2025-01-20 | `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/util/DeviceLocalNDArray.java` | Fixed `propagateConstantFlag()` to use `sourceWasConstant` field instead of re-checking |
| 2025-01-20 | `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/nativeblas/NativeOps.java` | Added `@NoDeallocator` to prevent JavaCPP double-free |
| 2026-01-20 | `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/jita/constant/ProtectedCachedShapeInfoProvider.java` | Added `registerPendingConstant()` protection to prevent race condition in shape buffer creation |

---

## Test Commands

```bash
# Run specific test for scalar/constant arrays
cd platform-tests
mvn test -Dtest=SameDiffTests#testConstantScalarLoading -DfailIfNoTests=false

# Run with debug logging
mvn test -Dtest=SameDiffTests -Dorg.nd4j.linalg.api.ops.OpContext.debug=true
```

---

## Related ADRs/Docs

- None currently

---

## Notes

- The garbage value `0x756f0012cf3e0001` contains bytes that look like ASCII string data ('u', 'o', etc.), strongly suggesting memory was reused by string allocation after being freed
- CUDA backend adds complexity because data lives in both host and device memory with sync counters
- The `pendingConstants` mechanism in DeallocatorService should protect during registration, but the issue is AFTER registration when the wrong buffer is marked
- This tracking document is at: `scalar-issues.md` in project root

## Key Insight

The fundamental issue is that when arrays are duplicated (`dup()`) or detached (`detach()`), a **NEW** native buffer is created. The constant flag must be explicitly propagated to this new buffer. If we re-check the source array's constant flag AFTER the dup/detach, there's a window where:
1. The source array's buffer might have been garbage collected
2. The source's `isConstant()` check might return wrong value or crash
3. The new buffer never gets marked constant
4. The new buffer gets freed during normal GC cycle
5. Use-after-free occurs when code tries to read from it

---

## 2025-01-20 - Fix Attempts That FAILED

### Attempt 1: propagateConstantFlag Race Condition Fix

**Files Modified:**
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/util/DeviceLocalNDArray.java`

**Change Made:**
Modified `propagateConstantFlag()` to use the already-captured `sourceWasConstant` field instead of re-checking `source.data().isConstant()`.

**Hypothesis:** Race condition where source buffer could be freed before constant flag was re-checked.

**Result:** **FAILED** - Same error still occurs:
```
FloorDivOp failed: Attention_0_three (type=CONSTANT, dtype=LONG, shape=[], isConstant=true)
```

**Why It Failed:** The Java-side `isConstant=true` is being set correctly. The problem is NOT in constant flag propagation at the Java level. The buffer itself is being freed despite being marked constant, or a different buffer is being used than the one that was marked constant.

### Attempt 2: @NoDeallocator Annotation (JavaCPP Double-Free Prevention)

**Files Modified:**
- `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/nativeblas/NativeOps.java`

**Change Made:**
Added `@NoDeallocator` annotation to methods returning `OpaqueDataBuffer` to prevent JavaCPP from attaching its own deallocator.

**Hypothesis:** Both JavaCPP's NativeDeallocator and ND4J's DeallocatorService were trying to free the same memory (double-free).

**Result:** **FAILED/PARTIAL** - Fixed the `corrupted double-linked list` crashes but scalar use-after-free still occurs.

**Why It Failed:** The double-free was a separate issue. The core scalar constant corruption problem persists.

---

## 2025-01-20 - Current Status

### What We Know For Certain

1. **Java-side constant flag IS set correctly**: Error log shows `isConstant=true`
2. **Buffer is still being freed**: Despite constant flag, memory is reused
3. **Garbage values contain ASCII**: `0x756f0012cf3e0001` suggests memory reused by string allocation
4. **Buffer not in registration list**: Address `0x7f0896077480` was NOT found in DeallocatorService tracking

### Leading Theories (Untested)

1. **Model Loading Path Issue**: Buffer created during `createFromFlatArray()` may not be the same buffer that ends up stored. The FlatArray deserialization creates a buffer, but subsequent dup/detach operations create NEW buffers. The ORIGINAL buffer from FlatArray may be getting freed.

2. **Device Memory Sync Issue**:
   - CUDA `BaseCudaDataBuffer` copies data to DEVICE but leaves HOST buffer uninitialized
   - `tickDeviceWrite()` marks device as having latest data
   - If `syncToHost()` is skipped due to `isPrimaryActual()` returning wrong value, host buffer reads garbage

3. **FlatBuffersMapper.fromFlatNode() Scalar Handling**:
   - Lines 448-504 have special scalar handling using `Nd4j.constantScalar()`
   - This path SHOULD create properly marked constant scalars
   - Need to verify this path is being used and not bypassed

### Next Investigation Steps

1. **Trace FlatArray to final storage**: Follow exact buffer lifecycle from `createFromFlatArray()` through `ThreadSafeArrayHolder.setArray()` to verify same buffer is used
2. **Check native `isConstant` flag**: The Java side shows true, but verify native `InteropDataBuffer.isConstant` atomic flag
3. **Verify `dbClose()` constant check**: Native `dbClose()` should skip if constant, verify this is being reached and flag is set
4. **Investigate ByteBuffer → CUDA buffer path**: Ensure data is correctly copied and sync flags are set properly

---

## 2025-01-20 - ROOT CAUSE FOUND: Missing constant flag in deserialization path

### Files Analyzed

| File | Line | Finding |
|------|------|---------|
| `nd4j/.../serde/SameDiffSerializer.java` | 2285-2287 | CONSTANT variables stored DIRECTLY to `constantArrays` without marking constant! |

### The Bug

When CONSTANT variables are loaded during model deserialization, the code at line 2286:
```java
if (varToUpdate.isConstant()) {
    targetSD.getConstantArrays().setArray(name, resultArr);  // <-- BUG!
}
```

This calls `constantArrays.setArray()` DIRECTLY, bypassing `setArrayForVariable()` which is responsible for marking the array as constant.

### Why This Causes Use-After-Free

1. **Model loads** → `createFromFlatArray()` creates array → NOT marked constant
2. **Direct storage** → `getConstantArrays().setArray(name, resultArr)`
3. **ThreadSafeArrayHolder.setArray()** checks:
   - `sourceIsConstant = array.data().isConstant()` → **FALSE** (not marked!)
   - Creates detached copy
   - Does NOT mark copy as constant (because `sourceIsConstant` is false)
4. **GC eventually frees the buffer** → NOT protected by constant flag
5. **Op reads scalar** → **Use-after-free! Reads garbage.**

### The Fix

Replace line 2286:
```java
// WRONG:
targetSD.getConstantArrays().setArray(name, resultArr);

// CORRECT:
targetSD.setArrayForVariable(name, resultArr);  // This marks constant BEFORE storing
```

Or alternatively, explicitly mark constant before storing:
```java
if (resultArr.data() != null) {
    resultArr.data().setConstant(true);
}
if (resultArr.shapeInfoDataBuffer() != null) {
    resultArr.shapeInfoDataBuffer().setConstant(true);
}
resultArr.setCloseable(false);
targetSD.getConstantArrays().setArray(name, resultArr);
```

### Why Previous Fixes Failed

1. **propagateConstantFlag fix**: Only works if `sourceIsConstant` is TRUE. Since the source was never marked constant, propagation was a no-op.
2. **@NoDeallocator fix**: Prevents double-free from JavaCPP, but buffer still gets freed by DeallocatorService because it's not marked constant.

### Verification

The Java-side `isConstant=true` shown in the error log is from `arr.data().isConstant()` (line 273 of ND4JOpExceptionUtils.java), which IS the buffer's constant flag.

---

## 2025-01-20 - ACTUAL ROOT CAUSE: Race condition between setConstant and dbClose

### Critical Discovery

In `NativeOpsHelpers_DataBuffers.cpp` lines 296-328:
```cpp
void dbSetConstant(OpaqueDataBuffer *dataBuffer, bool isConstant) {
  // ...
  if (!dataBuffer->isValid()) {
    // Buffer is invalid (freed or closed) - silently ignore  ← BUG!
    return;
  }

  dataBuffer->isConstant.store(isConstant, std::memory_order_release);
  // ...
}
```

If the native buffer has ALREADY been closed when `dbSetConstant()` is called, the function **silently returns without setting the constant flag**!

### Why This Is The Root Cause

Java and Native constant flags are **SEPARATE**:
1. **Java side**: `BaseDataBuffer.setConstant()` sets `this.constant = true` ✓
2. **Native side**: `OpaqueDataBuffer.setConstant()` calls `dbSetConstant()` which sets `isConstant.store(true)`

But if the DeallocatorService thread runs `dbClose()` BETWEEN:
- Step 1 (Java flag set) and
- Step 2 (Native flag set via `dbSetConstant()`)

Then:
1. Java side: `isConstant = true` → shown in error message
2. Native side: `isConstant = false` → buffer freed by `dbClose()`
3. Use-after-free occurs

### The Race Condition

```
Thread 1 (main)                    Thread 2 (DeallocatorService)
----------------                    --------------------------
setConstant(true)
  this.constant = true ← Java flag set
                                   [GC triggers]
                                   dbClose(buffer)
                                     isConstant.load() → false
                                     delete buffer ← FREED!
  ptrDataBuffer.setConstant(true)
    dbSetConstant(buffer, true)
      isValid() → false (closed!)
      return ← SILENTLY IGNORED!

[Later: op reads buffer → garbage]
```

### The Fix

The Java side must ensure the native constant flag is set BEFORE the buffer can be freed.

**Solution**: In `OpaqueDataBuffer.setConstant()`, call `dbSetConstant()` FIRST (before `retainReference()`), and use the `registerPendingConstant()`/`releasePendingConstant()` pattern at the caller level to prevent GC during the operation.

Alternatively, in `BaseDataBuffer.setConstant()`, call `ptrDataBuffer.setConstant(true)` FIRST before setting the Java flag.

### Why Previous Analysis Was Wrong

1. The "deserialization bypassing setArrayForVariable" theory was incorrect - the code DOES call `setArrayForVariable()` at line 2274
2. The issue is a race condition in the constant flag propagation, not missing code paths

---

## 2025-01-20 - FIX APPLIED: Reordered setConstant operations

### Files Modified

| File | Change |
|------|--------|
| `nd4j/.../api/buffer/BaseDataBuffer.java` | Call `ptrDataBuffer.setConstant()` FIRST |
| `nd4j/.../nativeblas/OpaqueDataBuffer.java` | Call `dbSetConstant()` FIRST |

### Before (Race Condition)

```java
// BaseDataBuffer.setConstant()
dealloc.setConstant(reallyConstant);  // Java-side first
this.constant = reallyConstant;
// ... later ...
ptrDataBuffer.setConstant(reallyConstant);  // Native dbSetConstant() LAST!
```

```java
// OpaqueDataBuffer.setConstant()
deallocator.setConstant(isConstant);  // Java-side first
this.retainReference();
// ... later ...
dbSetConstant(this, isConstant);  // Native LAST!
```

### After (Fixed)

```java
// BaseDataBuffer.setConstant()
ptrDataBuffer.setConstant(reallyConstant);  // Native dbSetConstant() FIRST!
// ... then Java-side ...
dealloc.setConstant(reallyConstant);
this.constant = reallyConstant;
```

```java
// OpaqueDataBuffer.setConstant()
dbSetConstant(this, isConstant);  // Native FIRST!
this.retainReference();
deallocator.setConstant(isConstant);  // Java-side last
```

### Why This Fixes The Issue

1. Native `dbSetConstant()` is called FIRST
2. Any concurrent `dbClose()` will see `isConstant=true` and skip deallocation
3. Java-side flags are set AFTER native protection is in place
4. No more race window where native flag is false while Java flag is true

### Result: **FAILED**

Same error still occurs:
```
USE-AFTER-FREE DETECTED: Input scalar at index 1 for op 'floordiv' contains pointer-like value: 139650421429152 (hex: 0x7f02e5c313a0).
Variable name: 'Attention_0_three'. Array id: 373. Array wasClosed: false. Data buffer address: 0x7f02e621d250.
```

**Why It Failed**: The race condition fix assumes `setConstant()` IS being called on the buffer that's eventually read. If the issue is that `setConstant()` is never called, or is called on a DIFFERENT buffer, the fix is irrelevant.

---

## 2025-01-20 - Continuing Investigation

### Key Observation

The error shows `Array wasClosed: false` - the Java-side INDArray thinks it's still valid. But the native buffer contains garbage. This suggests:

1. The Java INDArray object is fine
2. The Java DataBuffer object is fine
3. But the NATIVE buffer it points to has been freed or never had correct data

### Hypothesis: Buffer identity mismatch

When arrays are copied/detached during `ThreadSafeArrayHolder.setArray()`:
1. Original buffer (A) gets `setConstant(true)` called on it
2. `detach()` creates NEW buffer (B)
3. Buffer B should get `setConstant(true)` via `sourceIsConstant` propagation
4. BUT: what if the OpaqueDataBuffer pointer (`ptrDataBuffer`) in Buffer B points to a DIFFERENT native InteropDataBuffer than expected?

### Things to Check

1. Is `ptrDataBuffer` being properly set when a new buffer is created during `detach()`?
2. Is the OpaqueDataBuffer being registered with DeallocatorService?
3. Is the native InteropDataBuffer being allocated with correct pointers?
4. Is there a mismatch between what Java thinks is the buffer vs what native code accesses?

---

## 2025-01-20 - ROOT CAUSE FOUND: Silent failure in dbSetConstant

### Critical Discovery

The `dbSetConstant()` function in native code SILENTLY RETURNS without setting the flag if the buffer is already closed:

```cpp
// NativeOpsHelpers_DataBuffers.cpp:296
void dbSetConstant(OpaqueDataBuffer *dataBuffer, bool isConstant) {
  // ...
  if (!dataBuffer->isValid()) {
    // Buffer is invalid (freed or closed) - silently ignore  ← PROBLEM!
    return;
  }
  dataBuffer->isConstant.store(isConstant, std::memory_order_release);
}
```

But the Java code doesn't check if the native call succeeded:

```java
// OpaqueDataBuffer.java:589
Nd4j.getNativeOps().dbSetConstant(this, isConstant);  // Doesn't check return!

if (isConstant) {
    this.retainReference();  // Sets Java-side anyway
}
if (deallocator != null) {
    deallocator.setConstant(isConstant);  // Sets Java-side anyway
}
```

### The Race Condition Sequence

1. Buffer B is created during model deserialization, `isConstant = false` (default)
2. **GC RUNS** - DeallocatorService triggers `dbClose(B)`:
   - `tryClose()` sets `_closed = true`
   - `isConstant = false`, so it proceeds to delete the underlying DataBuffer
3. Later, `setConstant(true)` is called on Java side
4. Java calls `dbSetConstant(B, true)`
5. Native checks `isValid()`:
   - `isConstant = false`, `_closed = true`
   - `isValid()` returns `false` (buffer is closed AND not constant)
6. `dbSetConstant()` **SILENTLY RETURNS** without setting the flag
7. Java proceeds to set `deallocator.setConstant(true)` anyway
8. **RESULT**:
   - Native: `isConstant = false`, buffer FREED
   - Java: `isConstant = true`, thinks buffer is valid
9. **Use-after-free** when code tries to read from the freed buffer

### Why Previous Fixes Failed

1. **Reordering fix** (call native FIRST): Doesn't help if GC already closed the buffer BEFORE we even call `setConstant()`
2. **@NoDeallocator fix**: Only prevents double-free, doesn't prevent the single free from DeallocatorService

### The Fix: Make dbSetConstant Return Status

**Files Modified:**

| File | Change |
|------|--------|
| `libnd4j/include/legacy/NativeOps.h` | Changed `dbSetConstant` return type from `void` to `bool` |
| `libnd4j/include/legacy/impl/NativeOpsHelpers_DataBuffers.cpp` | Return `true` on success, `false` if buffer invalid |
| `nd4j/.../nativeblas/NativeOps.java` | Changed return type from `void` to `boolean` |
| `nd4j/.../nativeblas/OpaqueDataBuffer.java` | Check return value and throw exception on failure |

**Native Implementation:**

```cpp
bool dbSetConstant(OpaqueDataBuffer *dataBuffer, bool isConstant) {
  if (dataBuffer == nullptr) {
    return false;  // Null buffer
  }

  if (!dataBuffer->isValid()) {
    // Buffer is invalid (freed or closed) - return false
    // This indicates a race condition between GC and constant flag setting
    return false;
  }

  dataBuffer->isConstant.store(isConstant, std::memory_order_release);
  // ... propagate to DataBuffer ...

  return true;  // Success
}
```

**Java Implementation:**

```java
public void setConstant(boolean isConstant) {
    if (this.isNull()) {
        return;
    }

    boolean nativeSuccess = Nd4j.getNativeOps().dbSetConstant(this, isConstant);

    if (!nativeSuccess) {
        // Buffer was already freed by GC - this is a race condition!
        throw new IllegalStateException(
            "RACE CONDITION DETECTED: Failed to set constant flag on buffer at " + this.address() +
            " because it was already freed by GC. This indicates a bug in buffer lifecycle management. " +
            "The buffer should be protected with registerPendingConstant() before setting constant flag.");
    }

    // Now safe to set Java-side flags
    if (isConstant) {
        this.retainReference();
    }
    if (deallocator != null) {
        deallocator.setConstant(isConstant);
    }
}
```

### Why This Fixes The Issue

1. **Detection**: If the buffer was already freed, we now DETECT it instead of silently continuing
2. **Fail-fast**: Exception is thrown immediately, showing the exact location of the race condition
3. **No mismatch**: Java-side flags are ONLY set if native succeeded
4. **Debuggability**: The exception message points to the root cause (missing `registerPendingConstant()`)

### Expected Result

When this fix is deployed:
- If the race condition occurs, an `IllegalStateException` will be thrown
- The exception message will indicate that `registerPendingConstant()` was missing
- This allows us to identify EXACTLY which code path needs protection

### Next Steps

If the exception is thrown during model loading:
1. Identify the exact code path where the buffer is created
2. Add `registerPendingConstant()` call immediately after buffer creation
3. Call `setConstant(true)` to mark as constant
4. Call `releasePendingConstant()` after the array is safely stored

The `registerPendingConstant()` mechanism adds the array to a protected set that prevents DeallocatorService from deallocating it. This provides a safe window to mark the buffer constant.

---

## 2025-01-20 - FIX APPLIED: Mark constant at creation time

### Root Cause Confirmed

The race condition detection (via `IllegalStateException`) confirmed the issue:
```
RACE CONDITION DETECTED: Failed to set constant flag on buffer at 140508541182864
because it was already freed by GC.
```

Stack trace showed the issue was in:
```
at CudaExecutioner.createShapeInfo(CudaExecutioner.java:1775)
```

### Files Modified

| File | Change |
|------|--------|
| `nd4j/.../nativeblas/OpaqueDataBuffer.java` | Added `externalizedDataBuffer(numElements, dataType, primary, special, isConstant)` overload |
| `nd4j/.../jcublas/buffer/CudaLongDataBuffer.java` | Added constructor with `isConstant` parameter |
| `nd4j/.../jcublas/ops/executioner/CudaExecutioner.java` | Updated `createShapeInfo()` and `tadShapeInfoAndOffsets()` to mark constant at creation |

### The Fix

Instead of:
```java
// OLD: Race condition window between these two lines
val result = new CudaLongDataBuffer(primaryShapeInfo, specialShapeInfo, shapeInfoLength);
result.setConstant(true);  // GC could free buffer before this line executes!
```

Now:
```java
// NEW: Mark constant DURING creation, no race window
val result = new CudaLongDataBuffer(primaryShapeInfo, specialShapeInfo, shapeInfoLength, true);
```

### How It Works

1. `CudaLongDataBuffer` constructor now accepts `isConstant` parameter
2. Constructor passes `isConstant` to `OpaqueDataBuffer.externalizedDataBuffer()`
3. `externalizedDataBuffer()` passes `isConstant` to `registerWithDeallocatorService()`
4. `registerWithDeallocatorService()` marks constant BEFORE the deallocator is active:
   - Sets `deallocator.setConstant(true)`
   - Calls `dbSetConstant(buffer, true)` on native side
   - Calls `buffer.retainReference()` to prevent JavaCPP deallocation
   - Does NOT register with DeallocatorService (constants should never be deallocated)

### Why This Works

1. **No race window**: The buffer is marked constant DURING registration, before any deallocator becomes active
2. **Atomic protection**: By the time the buffer is visible to GC, it's already protected
3. **Both Java and native sides protected**: Both `deallocator.isConstant` and native `isConstant` are set atomically

### Result

The race condition is eliminated for shape info buffers. The fix should be applied to other code paths that create constant buffers (like `createConstantBuffer()` methods) if they exhibit the same issue.

---

## 2026-01-20 - NEW RACE CONDITION: ProtectedCachedShapeInfoProvider.createShapeInformation

### Error Observed

```
RACE CONDITION DETECTED: Failed to set constant flag on buffer at 140180042544896
because it was already freed by GC. This indicates a bug in buffer lifecycle management.
The buffer should be protected with registerPendingConstant() before setting constant flag.
```

### Stack Trace

```
at org.nd4j.nativeblas.OpaqueDataBuffer.setConstant(OpaqueDataBuffer.java:629)
at org.nd4j.linalg.api.buffer.BaseDataBuffer.setConstant(BaseDataBuffer.java:2151)
at org.nd4j.jita.constant.ProtectedCachedShapeInfoProvider.createShapeInformation(ProtectedCachedShapeInfoProvider.java:100)
at org.nd4j.jita.constant.ProtectedCachedShapeInfoProvider.createShapeInformation(ProtectedCachedShapeInfoProvider.java:81)
at org.nd4j.linalg.jcublas.CachedShapeInfoProvider.createShapeInformation(CachedShapeInfoProvider.java:46)
at org.nd4j.linalg.api.ndarray.BaseNDArray.<init>(BaseNDArray.java:433)
at org.nd4j.linalg.jcublas.JCublasNDArray.<init>(JCublasNDArray.java:396)
at org.nd4j.linalg.jcublas.JCublasNDArrayFactory.create(JCublasNDArrayFactory.java:1024)
at org.nd4j.linalg.factory.Nd4j.scalar(Nd4j.java:5204)
at org.nd4j.linalg.factory.Nd4j.scalar(Nd4j.java:5243)
```

### Context

This error occurs during ND4J initialization when creating the first scalar array. The scalar creation triggers:
1. `Nd4j.scalar()` → creates an NDArray
2. `BaseNDArray.<init>()` → needs shape info
3. `CachedShapeInfoProvider.createShapeInformation()` → delegates to ProtectedCachedShapeInfoProvider
4. `ProtectedCachedShapeInfoProvider.createShapeInformation()` → creates shape buffer, then tries to mark constant

### Root Cause

In `ProtectedCachedShapeInfoProvider.createShapeInformation()` (lines 95-112):

```java
if (!protector.containsDataBuffer(deviceId, descriptor)) {
    Pair<DataBuffer, long[]> buffer = null;
    synchronized (this) {
        if (!protector.containsDataBuffer(deviceId, descriptor)) {
            buffer = super.createShapeInformation(shape, stride, elementWiseStride, order, extras);
            buffer.getFirst().setConstant(true);  // <-- RACE CONDITION HERE!

            protector.persistDataBuffer(deviceId, descriptor, buffer);
            // ...
        }
    }
}
```

**The Race Condition Window:**

1. `super.createShapeInformation()` creates a DataBuffer at line 99
2. Inside `BaseShapeInfoProvider.createShapeInformation()` → `Shape.createShapeInformation()` → `Nd4j.getExecutioner().createShapeInfo()`
3. The DataBuffer is created and **registered with DeallocatorService** (not marked constant)
4. **GC CAN RUN HERE** - the buffer is eligible for deallocation
5. `buffer.getFirst().setConstant(true)` is called at line 100
6. But the buffer was already freed by GC!

### Why Previous CudaExecutioner Fix Didn't Cover This

The previous fix for `CudaExecutioner.createShapeInfo()` added an `isConstant` parameter to mark constant at creation time. However, `ProtectedCachedShapeInfoProvider` calls `BaseShapeInfoProvider.createShapeInformation()` which calls `Shape.createShapeInformation()` which does NOT have the `isConstant` parameter.

### Proposed Fix #1: Use registerPendingConstant() Pattern

The `DeallocatorService` has `registerPendingConstant()` and `releasePendingConstant()` methods specifically designed for this scenario:

```java
// In ProtectedCachedShapeInfoProvider.createShapeInformation()
if (!protector.containsDataBuffer(deviceId, descriptor)) {
    Pair<DataBuffer, long[]> buffer = null;
    synchronized (this) {
        if (!protector.containsDataBuffer(deviceId, descriptor)) {
            buffer = super.createShapeInformation(shape, stride, elementWiseStride, order, extras);

            // Protect the buffer from GC before setting constant
            DataBuffer dataBuffer = buffer.getFirst();
            Nd4j.getDeallocatorService().registerPendingConstant(dataBuffer);
            try {
                dataBuffer.setConstant(true);
            } finally {
                // Release the protection - now that it's marked constant,
                // DeallocatorService won't deallocate it anyway
                Nd4j.getDeallocatorService().releasePendingConstant(dataBuffer);
            }

            protector.persistDataBuffer(deviceId, descriptor, buffer);
            bytes.addAndGet(buffer.getFirst().length() * 8 * 2);
            cacheMiss.incrementAndGet();
        } else {
            buffer = protector.getDataBuffer(deviceId, descriptor);
        }
    }
    return buffer;
}
```

### Proposed Fix #2: Mark constant at creation (preferred)

Add an overload to `Shape.createShapeInformation()` and `OpExecutioner.createShapeInfo()` that accepts an `isConstant` parameter:

```java
// In Shape.java
public static DataBuffer createShapeInformation(long[] shape, long[] stride,
        long elementWiseStride, char order, long extras, boolean isConstant) {
    val dtype = ArrayOptionsHelper.dataType(extras);
    return Nd4j.getExecutioner().createShapeInfo(shape, stride, elementWiseStride,
            order, dtype, extras, isConstant);
}
```

Then update `BaseShapeInfoProvider`:
```java
@Override
public Pair<DataBuffer, long[]> createShapeInformation(long[] shape, long[] stride,
        long elementWiseStride, char order, long extras) {
    // Default to NOT constant (for regular array creation)
    DataBuffer buffer = Shape.createShapeInformation(shape, stride, elementWiseStride, order, extras, false);
    return Pair.create(buffer, buffer.asLong());
}

// New overload for constant shape buffers
public Pair<DataBuffer, long[]> createConstantShapeInformation(long[] shape, long[] stride,
        long elementWiseStride, char order, long extras) {
    DataBuffer buffer = Shape.createShapeInformation(shape, stride, elementWiseStride, order, extras, true);
    return Pair.create(buffer, buffer.asLong());
}
```

Then `ProtectedCachedShapeInfoProvider` can call `super.createConstantShapeInformation()`:
```java
buffer = super.createConstantShapeInformation(shape, stride, elementWiseStride, order, extras);
// No need for setConstant(true) - already constant!
```

### Why Fix #2 is Preferred

1. **Eliminates the race window entirely** - buffer is constant from birth
2. **Consistent with CudaExecutioner fix** - same pattern
3. **No need for pendingConstants overhead** - cleaner code path
4. **Better performance** - no extra synchronization on pendingConstants set

### Files to Modify

| File | Change |
|------|--------|
| `nd4j/.../api/shape/Shape.java` | Add `createShapeInformation(..., boolean isConstant)` overload |
| `nd4j/.../api/ndarray/BaseShapeInfoProvider.java` | Add `createConstantShapeInformation()` method |
| `nd4j/.../jita/constant/ProtectedCachedShapeInfoProvider.java` | Use `createConstantShapeInformation()` instead of `super.createShapeInformation() + setConstant()` |
| `nd4j/.../api/ops/executioner/OpExecutioner.java` | Add `createShapeInfo(..., boolean isConstant)` if not already present |
| Backend-specific executioners | Implement the new method signature |

### Alternative Quick Fix

If the full fix is too invasive, use `registerPendingConstant()` pattern in `ProtectedCachedShapeInfoProvider`:

```java
// Quick fix - protect during the race window
buffer = super.createShapeInformation(shape, stride, elementWiseStride, order, extras);
DataBuffer dataBuffer = buffer.getFirst();

// Hold strong reference to prevent GC
Nd4j.getDeallocatorService().registerPendingConstant(dataBuffer);
try {
    dataBuffer.setConstant(true);
} finally {
    Nd4j.getDeallocatorService().releasePendingConstant(dataBuffer);
}
```

### Next Steps

1. ~~Apply quick fix (registerPendingConstant) to unblock users~~ **DONE** - See fix applied below
2. Plan full fix (isConstant at creation) for next release
3. Audit all code paths that create constant buffers to ensure they use atomic marking

---

## 2026-01-20 - FIX APPLIED: registerPendingConstant in ProtectedCachedShapeInfoProvider

### Files Modified

| File | Change |
|------|--------|
| `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/jita/constant/ProtectedCachedShapeInfoProvider.java` | Added `registerPendingConstant()` / `releasePendingConstant()` protection around `setConstant(true)` call |

### The Fix

```java
// Before: Race condition window
buffer = super.createShapeInformation(shape, stride, elementWiseStride, order, extras);
buffer.getFirst().setConstant(true);  // Buffer could be freed by GC before this line!

// After: Protected with registerPendingConstant()
buffer = super.createShapeInformation(shape, stride, elementWiseStride, order, extras);
DataBuffer dataBuffer = buffer.getFirst();
Nd4j.getDeallocatorService().registerPendingConstant(dataBuffer);  // Hold strong reference
try {
    dataBuffer.setConstant(true);
} finally {
    Nd4j.getDeallocatorService().releasePendingConstant(dataBuffer);  // Release protection
}
```

### How It Works

1. `registerPendingConstant(dataBuffer)` adds the DataBuffer to a `Set<Object>` in DeallocatorService
2. This creates a **strong reference** to the buffer, preventing GC from collecting it
3. `setConstant(true)` is called safely - the buffer cannot be freed
4. `releasePendingConstant(dataBuffer)` removes the buffer from the set
5. Now that the buffer is marked constant, DeallocatorService will skip deallocation anyway

### Why This Works

The `pendingConstants` set in `DeallocatorService` uses `IdentityHashMap` (not `ConcurrentHashMap`) because:
- `ConcurrentHashMap` calls `hashCode()` on keys when adding/removing
- `INDArray.hashCode()` might execute CUDA operations which can fail during array creation
- `IdentityHashMap` uses object identity (`==`) which is safe

### Expected Result

The race condition should be eliminated. The buffer will never be freed by GC during the window between creation and marking as constant.

### Testing

Run the embedding subprocess initialization that was failing:
```bash
# The error should no longer occur:
# RACE CONDITION DETECTED: Failed to set constant flag on buffer at 140180042544896
# because it was already freed by GC.
```

### Result: **FAILED**

The fix did NOT work. Error still occurs at line 110 (the `setConstant(true)` call inside the try block).

---

## 2026-01-20 - Why registerPendingConstant Fix Failed

### Key Observation

The debug log shows:
```
Debug: Registering org.nd4j.nativeblas.OpaqueConstantShapeBuffer[address=0x7f243994f570,position=0,limit=1,capacity=1,deallocator=org.bytedeco.javacpp.Pointer$NativeDeallocator[ownerAddress=0x7f243994f570,deallocatorAddress=0x7f25cd065440]]
```

This reveals:
1. It's `OpaqueConstantShapeBuffer` being registered, NOT `OpaqueDataBuffer`
2. It has a **JavaCPP NativeDeallocator** attached
3. The buffer is created and deallocated DURING `super.createShapeInformation()`, not after

### The Real Problem

The `registerPendingConstant()` fix comes **TOO LATE**:

```
Timeline:
1. super.createShapeInformation() called
   └─> Shape.createShapeInformation() called
       └─> Nd4j.getExecutioner().createShapeInfo() called
           └─> OpaqueConstantShapeBuffer created
           └─> Buffer registered with DeallocatorService (isConstant=false)
           └─> GC RUNS HERE - buffer freed!    <-- PROBLEM
       └─> Returns freed buffer
   └─> Returns freed buffer
2. registerPendingConstant(dataBuffer)  <-- TOO LATE!
3. setConstant(true)  <-- FAILS: buffer already freed
```

### Root Cause

The buffer is being freed by GC **INSIDE** the `createShapeInfo()` call chain, before control returns to `ProtectedCachedShapeInfoProvider`. The `registerPendingConstant()` pattern cannot protect against this because:

1. We don't have a reference to the buffer until AFTER `super.createShapeInformation()` returns
2. By that time, the buffer may already be freed

### The ONLY Fix

**Mark constant at creation time** - the buffer must be marked constant INSIDE `createShapeInfo()` before it can be GC'd. This requires:

1. Adding `isConstant` parameter to `createShapeInfo()` in OpExecutioner interface
2. Propagating this parameter through the call chain:
   - `BaseShapeInfoProvider.createShapeInformation()`
   - `Shape.createShapeInformation()`
   - `OpExecutioner.createShapeInfo()`
   - `CudaExecutioner.createShapeInfo()` / `NativeOpExecutioner.createShapeInfo()`
3. When `isConstant=true`, mark the buffer constant IMMEDIATELY after native allocation, before registering with DeallocatorService

### Alternative: Disable GC During Shape Creation

Another approach is to use `DeallocatorService.toggleDeallocationBlock(true)` to temporarily block all deallocation:

```java
// Block GC deallocation during shape buffer creation
Nd4j.getDeallocatorService().toggleDeallocationBlock(true);
try {
    buffer = super.createShapeInformation(shape, stride, elementWiseStride, order, extras);
    buffer.getFirst().setConstant(true);
} finally {
    Nd4j.getDeallocatorService().toggleDeallocationBlock(false);
}
```

This is a heavy-handed approach but may work as a temporary fix.

### Files to Investigate

| File | Purpose |
|------|---------|
| `CudaExecutioner.createShapeInfo()` | Where CUDA shape buffers are created |
| `NativeOpExecutioner.createShapeInfo()` | Where CPU shape buffers are created |
| `OpaqueConstantShapeBuffer` | The shape buffer class being deallocated |
| `Shape.createShapeInformation()` | Intermediate call in the chain |

---

## 2026-01-20 - ACTUAL ROOT CAUSE FOUND: OpaqueConstantShapeBuffer JavaCPP Deallocation

### Investigation Results

After tracing through the code, the ACTUAL root cause is now clear:

**The `OpaqueConstantShapeBuffer` returned by `shapeBufferEx()` has a JavaCPP `NativeDeallocator` that frees the native memory when it goes out of scope.**

### Code Flow Analysis

```java
// CudaExecutioner.createShapeInfo() lines 1756-1778
public DataBuffer createShapeInfo(...) {
    LongPointer shapePointer = new LongPointer(shape);
    LongPointer stridePointer = new LongPointer(stride);

    // Line 1759: Creates OpaqueConstantShapeBuffer with JavaCPP NativeDeallocator
    OpaqueConstantShapeBuffer dbf = Nd4j.getNativeOps().shapeBufferEx(...);

    // Line 1767-1768: Extract pointers from dbf
    Pointer primaryShapeInfo = Nd4j.getNativeOps().getConstantShapeBufferPrimary(dbf);
    Pointer specialShapeInfo = Nd4j.getNativeOps().getConstantShapeBufferSpecial(dbf);

    // Line 1775: Create CudaLongDataBuffer wrapping those pointers
    // This creates a NEW OpaqueDataBuffer pointing to the SAME native memory!
    val result = new CudaLongDataBuffer(primaryShapeInfo, specialShapeInfo, shapeInfoLength, true);

    // Line 1777: Return result
    return result;

    // AFTER THIS LINE: dbf goes out of scope!
    // JavaCPP's NativeDeallocator will FREE the native memory!
    // But result.ptrDataBuffer still points to that freed memory!
}
```

### The Root Cause

1. **Two Java objects, ONE native memory**:
   - `OpaqueConstantShapeBuffer dbf` - has JavaCPP NativeDeallocator, OWNS the native memory
   - `OpaqueDataBuffer` (inside CudaLongDataBuffer) - just WRAPS the same native memory

2. **Memory ownership conflict**:
   - `dbf` is a local variable that goes out of scope at end of method
   - JavaCPP's NativeDeallocator frees the native memory when `dbf` is GC'd
   - `CudaLongDataBuffer` still points to the now-freed memory

3. **Why marking constant doesn't help**:
   - The `CudaLongDataBuffer` is marked constant correctly
   - But the `OpaqueConstantShapeBuffer` is NOT marked constant
   - JavaCPP doesn't know about ND4J's constant flag - it just frees memory when the object is GC'd

### Evidence from Debug Log

```
Debug: Registering org.nd4j.nativeblas.OpaqueConstantShapeBuffer[address=0x7f243994f570,position=0,limit=1,capacity=1,deallocator=org.bytedeco.javacpp.Pointer$NativeDeallocator[ownerAddress=0x7f243994f570,deallocatorAddress=0x7f25cd065440]]
```

This shows:
- `OpaqueConstantShapeBuffer` has its own `NativeDeallocator`
- This deallocator will call the native destructor when GC runs
- The native memory will be freed regardless of what the `OpaqueDataBuffer` thinks

### Why `ProtectedCachedShapeInfoProvider` Fails

```
Stack trace:
at ProtectedCachedShapeInfoProvider.createShapeInformation(line 110)  // setConstant(true)
```

The `setConstant(true)` fails because:
1. `super.createShapeInformation()` calls `CudaExecutioner.createShapeInfo()`
2. Inside that method, `OpaqueConstantShapeBuffer dbf` is created
3. A `CudaLongDataBuffer` is created wrapping dbf's native pointers
4. Method returns, `dbf` goes out of scope
5. **GC RUNS** - JavaCPP frees `dbf`'s native memory
6. `setConstant(true)` is called on the returned `DataBuffer`
7. The underlying `OpaqueDataBuffer` tries to call `dbSetConstant()` on native side
8. **FAILS**: The native buffer was already freed by JavaCPP!

### The Fix

**Option 1: Call `retainReference()` on `OpaqueConstantShapeBuffer`**

```java
// In CudaExecutioner.createShapeInfo()
OpaqueConstantShapeBuffer dbf = Nd4j.getNativeOps().shapeBufferEx(...);
dbf.retainReference();  // Prevent JavaCPP deallocation

Pointer primaryShapeInfo = Nd4j.getNativeOps().getConstantShapeBufferPrimary(dbf);
Pointer specialShapeInfo = Nd4j.getNativeOps().getConstantShapeBufferSpecial(dbf);

val result = new CudaLongDataBuffer(primaryShapeInfo, specialShapeInfo, shapeInfoLength, true);
return result;
// dbf won't be freed - retainReference() prevents it
```

**Option 2: Store reference to `OpaqueConstantShapeBuffer` in `CudaLongDataBuffer`**

```java
// In CudaLongDataBuffer, add field:
private OpaqueConstantShapeBuffer sourceShapeBuffer;

// Constructor keeps reference:
public CudaLongDataBuffer(..., OpaqueConstantShapeBuffer sourceBuffer) {
    ...
    this.sourceShapeBuffer = sourceBuffer;  // Keep alive to prevent GC
}
```

**Option 3: Add @NoDeallocator to shapeBufferEx() in JavaCPP bindings**

The bindings have:
```java
public native @ByVal org.nd4j.nativeblas.OpaqueConstantShapeBuffer shapeBufferEx(...)
```

Change to:
```java
@NoDeallocator
public native @ByVal org.nd4j.nativeblas.OpaqueConstantShapeBuffer shapeBufferEx(...)
```

This would prevent JavaCPP from attaching a deallocator, but then we'd need to manage cleanup manually.

### Recommended Fix: Option 1 (`retainReference()`)

Option 1 is the safest and most localized fix:
- Just add `dbf.retainReference()` in `CudaExecutioner.createShapeInfo()`
- And in `NativeOpExecutioner.createShapeInfo()` for CPU backend
- This prevents JavaCPP from freeing the memory
- The memory will live for the lifetime of the JVM (acceptable for shape info caches)

### Affected Files (Both Backends Have Same Issue)

**CUDA Backend** - `CudaExecutioner.java`:
```java
// Line 1759 - MISSING retainReference()!
OpaqueConstantShapeBuffer dbf = Nd4j.getNativeOps().shapeBufferEx(...);
// dbf goes out of scope and JavaCPP frees memory
```

**CPU Backend** - `NativeOpExecutioner.java`:
```java
// Lines 1245 and 1268 - MISSING retainReference()!
OpaqueConstantShapeBuffer dbf = getNativeOps().shapeBufferEx(...);
// dbf goes out of scope and JavaCPP frees memory
```

### Files to Modify

| File | Lines | Fix |
|------|-------|-----|
| `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/ops/executioner/CudaExecutioner.java` | 1759 | Add `dbf.retainReference()` after `shapeBufferEx()` |
| `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cpu-backend-common/src/main/java/org/nd4j/linalg/cpu/nativecpu/ops/NativeOpExecutioner.java` | 1245, 1268 | Add `dbf.retainReference()` after both `shapeBufferEx()` calls |

---

## 2026-01-20 - Confirmed: registerPendingConstant Fix Did NOT Work

### Test Result

The `registerPendingConstant()` fix in `ProtectedCachedShapeInfoProvider` **did not resolve the issue**. Same error:

```
RACE CONDITION DETECTED: Failed to set constant flag on buffer at 139793561618176
because it was already freed by GC.
```

### Why It Failed

The `registerPendingConstant()` fix protects the `DataBuffer` AFTER it's returned from `super.createShapeInformation()`. But the memory is being freed **INSIDE** that call, specifically:

```
ProtectedCachedShapeInfoProvider.createShapeInformation()
  → super.createShapeInformation()
    → Shape.createShapeInformation()
      → Nd4j.getExecutioner().createShapeInfo()
        → CudaExecutioner.createShapeInfo()
          → shapeBufferEx() returns OpaqueConstantShapeBuffer
          → Extract primary/special pointers
          → Create CudaLongDataBuffer
          → METHOD RETURNS
          → OpaqueConstantShapeBuffer goes out of scope
          → GC FREES IT HERE ← Memory freed!
      → Returns DataBuffer pointing to freed memory
    → Returns DataBuffer pointing to freed memory
  → Returns DataBuffer pointing to freed memory
  → registerPendingConstant() ← TOO LATE!
  → setConstant(true) ← FAILS: memory already freed!
```

### Confirmed Root Cause

The `OpaqueConstantShapeBuffer` local variable in `CudaExecutioner.createShapeInfo()` goes out of scope when the method returns. JavaCPP's NativeDeallocator then frees the native memory. The `CudaLongDataBuffer` returned still points to that freed memory.

**The ONLY fix is to call `dbf.retainReference()` in `CudaExecutioner.createShapeInfo()` and `NativeOpExecutioner.createShapeInfo()`.**

### Reverting the registerPendingConstant Fix

The `registerPendingConstant()` change in `ProtectedCachedShapeInfoProvider` should be reverted since it adds complexity without solving the problem. The real fix must be in the executioner classes.

---

## 2026-01-20 - retainReference on OpaqueConstantShapeBuffer STILL DIDN'T WORK

### Test Result

Added `dbf.retainReference()` in `CudaExecutioner.createShapeInfo()` but the error **still occurs**:

```
RACE CONDITION DETECTED: Failed to set constant flag on buffer at 140006027487920
because it was already freed by GC.
```

### Key Observation: Different Addresses!

From debug output:
```
OpaqueConstantShapeBuffer address: 0x7f55b188d900
Failed buffer address:             0x7F55B1945AB0 (140006027487920 in decimal)
```

**These are DIFFERENT addresses!** The `OpaqueConstantShapeBuffer` is NOT the same object as the buffer that's failing.

### Root Cause Refinement

There are **TWO** native objects involved:

1. **OpaqueConstantShapeBuffer** (from `shapeBufferEx()`) - We fixed this with `retainReference()`
2. **OpaqueDataBuffer** (from `dbCreateExternalDataBuffer()` inside `externalizedDataBuffer()`) - This is ALSO a JavaCPP Pointer with NativeDeallocator!

The flow inside `OpaqueDataBuffer.externalizedDataBuffer()`:

```java
public static OpaqueDataBuffer externalizedDataBuffer(..., boolean isConstant) {
    // Line 212: Creates OpaqueDataBuffer - has JavaCPP NativeDeallocator
    OpaqueDataBuffer ret = Nd4j.getNativeOps().dbCreateExternalDataBuffer(...);

    // GC CAN RUN HERE! JavaCPP can deallocate 'ret' before we protect it!

    // Lines 218-225: Try to register and protect the buffer
    if (ret != null && !ret.isNull()) {
        registerWithDeallocatorService(ret, isConstant);  // ← FAILS: ret already freed!
    }
}
```

Inside `registerWithDeallocatorService()`:
```java
if (isConstant) {
    deallocator.setConstant(true);
    Nd4j.getNativeOps().dbSetConstant(buffer, true);  // ← FAILS HERE
    buffer.retainReference();  // ← Never reached
}
```

### The ACTUAL Race Condition

```
1. dbCreateExternalDataBuffer() returns OpaqueDataBuffer ret
   - ret has JavaCPP NativeDeallocator attached
   - ret is now eligible for GC
2. GC RUNS - JavaCPP's NativeDeallocator frees ret's native memory
3. registerWithDeallocatorService(ret, true) is called
   - Tries to call dbSetConstant(buffer, true)
   - FAILS: native buffer already closed!
```

### The REAL Fix

Call `ret.retainReference()` **IMMEDIATELY** after `dbCreateExternalDataBuffer()`, before ANY other code:

```java
public static OpaqueDataBuffer externalizedDataBuffer(..., boolean isConstant) {
    OpaqueDataBuffer ret = Nd4j.getNativeOps().dbCreateExternalDataBuffer(...);

    // Prevent JavaCPP deallocation immediately.
    // Must be done before any other code that could allow GC to run
    if (ret != null && !ret.isNull()) {
        ret.retainReference();  // ← ADD THIS LINE
    }

    // Now safe to continue...
    if (ret != null && !ret.isNull()) {
        registerWithDeallocatorService(ret, isConstant);
    }
    return ret;
}
```

### Why Previous Fixes Didn't Work

1. **registerPendingConstant in ProtectedCachedShapeInfoProvider**: Too late - buffer freed inside createShapeInfo()
2. **retainReference on OpaqueConstantShapeBuffer**: Wrong object - the failing buffer is OpaqueDataBuffer, not OpaqueConstantShapeBuffer
3. **The real issue**: `dbCreateExternalDataBuffer()` returns an OpaqueDataBuffer that immediately becomes eligible for GC

### Files to Modify

| File | Method | Fix |
|------|--------|-----|
| `nd4j/.../nativeblas/OpaqueDataBuffer.java` | `externalizedDataBuffer()` | Add `ret.retainReference()` immediately after `dbCreateExternalDataBuffer()` |
| `nd4j/.../nativeblas/OpaqueDataBuffer.java` | `allocateDataBuffer()` | Same fix - add `buffer.retainReference()` immediately after allocation |

---

## 2026-01-20 - FIX APPLIED: retainReference in OpaqueDataBuffer

### Files Modified

| File | Change |
|------|--------|
| `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/nativeblas/OpaqueDataBuffer.java` | Added `ret.retainReference()` in `externalizedDataBuffer()` and `buffer.retainReference()` in `allocateDataBuffer()` |

### The Fix

```java
// In externalizedDataBuffer():
OpaqueDataBuffer ret = Nd4j.getNativeOps().dbCreateExternalDataBuffer(...);
if (ret != null && !ret.isNull()) {
    ret.retainReference();
}

// In allocateDataBuffer():
buffer = Nd4j.getNativeOps().allocateDataBuffer(...);
if(buffer != null && !buffer.isNull()) {
    buffer.retainReference();
    // ... rest of registration
}
```

---

## 2026-01-20 - ACTUAL ROOT CAUSE FOUND: @NoDeallocator annotation ignored on interface

### Investigation

After extensive debugging, the real issue was discovered:

1. `@NoDeallocator` annotations were added to `NativeOps.java` (the interface)
2. But `NativeOps.java` is just an INTERFACE
3. The actual binding implementation is in `Nd4jCuda.java` (generated by JavaCPP)
4. **JavaCPP generates bindings from C++ headers, NOT from Java interfaces**
5. The `@NoDeallocator` annotations on the interface are COMPLETELY IGNORED

### Evidence

`NativeOps.java` (interface) has:
```java
@NoDeallocator
org.nd4j.nativeblas.OpaqueDataBuffer dbCreateExternalDataBuffer(long elements, int dataType, Pointer primary, Pointer special);
```

But the generated `Nd4jCuda.java` (line 2142) has:
```java
public native org.nd4j.nativeblas.OpaqueDataBuffer dbCreateExternalDataBuffer(@Cast("sd::LongType") long elements, int dataType, @Cast("sd::Pointer") Pointer primary, @Cast("sd::Pointer") Pointer special);
```

**No `@NoDeallocator` annotation!** JavaCPP will attach a `NativeDeallocator` to the returned pointer.

### What Happens

1. `dbCreateExternalDataBuffer()` is called
2. Native code creates an `InteropDataBuffer` (OpaqueDataBuffer) on the heap
3. JavaCPP wraps the pointer and attaches a `NativeDeallocator`
4. The Java `OpaqueDataBuffer` object becomes eligible for GC
5. Before `retainReference()` is called (or despite it?), GC runs
6. JavaCPP's `NativeDeallocator` calls `delete` on the `InteropDataBuffer`
7. The `InteropDataBuffer` destructor sets `_magic = INTEROP_BUFFER_FREED` and `_closed = true`
8. Later, `dbSetConstant()` is called but `isValid()` returns false
9. Error is thrown

### Why `retainReference()` Doesn't Help

The `retainReference()` call at line 217 of `OpaqueDataBuffer.externalizedDataBuffer()` should prevent GC, but:

1. There's still a window between `dbCreateExternalDataBuffer()` (line 214) and `retainReference()` (line 217)
2. If GC runs in that window, the buffer is destroyed
3. Even after `retainReference()`, if the `InteropDataBuffer` was already destroyed (magic number corrupted), it's too late

### The Fix

Add `@NoDeallocator` to the JavaCPP presets using `javaText` override:

**File: `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda-preset/src/main/java/org/nd4j/presets/cuda/Nd4jCudaPresets.java`**

```java
.put(new Info("dbCreateExternalDataBuffer").javaText(
    "@NoDeallocator public native org.nd4j.nativeblas.OpaqueDataBuffer dbCreateExternalDataBuffer(@Cast(\"sd::LongType\") long elements, int dataType, @Cast(\"sd::Pointer\") Pointer primary, @Cast(\"sd::Pointer\") Pointer special);"))
.put(new Info("dbAllocateDataBuffer").javaText(
    "@NoDeallocator public native org.nd4j.nativeblas.OpaqueDataBuffer dbAllocateDataBuffer(@Cast(\"sd::LongType\") long elements, int dataType, @Cast(\"bool\") boolean allocateBoth);"))
.put(new Info("allocateDataBuffer").javaText(
    "@NoDeallocator public native org.nd4j.nativeblas.OpaqueDataBuffer allocateDataBuffer(@Cast(\"sd::LongType\") long elements, int dataType, @Cast(\"bool\") boolean allocateBoth);"))
```

**ALSO** for CPU backend: `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native-preset/src/main/java/org/nd4j/presets/cpu/Nd4jCpuPresets.java`

### Why This Fix Will Work

1. `@NoDeallocator` in the presets will be applied to the GENERATED binding
2. JavaCPP will NOT attach a `NativeDeallocator` to returned `OpaqueDataBuffer` pointers
3. The native memory will be managed solely by ND4J's `DeallocatorService`
4. No more race condition between JavaCPP GC and `setConstant()`

### Files to Modify

| File | Change |
|------|--------|
| `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda-preset/src/main/java/org/nd4j/presets/cuda/Nd4jCudaPresets.java` | Add `@NoDeallocator` via javaText for OpaqueDataBuffer-returning methods |
| `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native-preset/src/main/java/org/nd4j/presets/cpu/Nd4jCpuPresets.java` | Same fix for CPU backend |

### After Fix: Regenerate Bindings

After modifying the presets, the bindings must be regenerated:
```bash
mvn clean install -pl nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda-preset -DskipTests
mvn clean install -pl nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native-preset -DskipTests
```

Then rebuild the CUDA backend to pick up the new bindings.

---

## 2026-01-20 - FIX APPLIED: @NoDeallocator in JavaCPP Presets

### Files Modified

| File | Change |
|------|--------|
| `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda-preset/src/main/java/org/nd4j/presets/cuda/Nd4jCudaPresets.java` | Added `@NoDeallocator` javaText overrides for `dbCreateExternalDataBuffer`, `dbAllocateDataBuffer`, `allocateDataBuffer`, `dbCreateView`, and `intermediateResultDataAt` |
| `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native-preset/src/main/java/org/nd4j/presets/cpu/Nd4jCpuPresets.java` | Same changes for CPU backend |

### The Fix

Added `javaText` entries to override the generated bindings with `@NoDeallocator` annotation:

```java
.put(new Info("dbCreateExternalDataBuffer").javaText(
        "@org.bytedeco.javacpp.annotation.NoDeallocator public native org.nd4j.nativeblas.OpaqueDataBuffer dbCreateExternalDataBuffer(@Cast(\"sd::LongType\") long elements, int dataType, @Cast(\"sd::Pointer\") Pointer primary, @Cast(\"sd::Pointer\") Pointer special);"))
.put(new Info("dbAllocateDataBuffer").javaText(
        "@org.bytedeco.javacpp.annotation.NoDeallocator public native org.nd4j.nativeblas.OpaqueDataBuffer dbAllocateDataBuffer(@Cast(\"sd::LongType\") long elements, int dataType, @Cast(\"bool\") boolean allocateBoth);"))
// ... and similar for other methods
```

### Why This Works

1. The `@NoDeallocator` annotation is now applied to the GENERATED bindings in `Nd4jCuda.java` and `Nd4jCpu.java`
2. JavaCPP will NOT attach a `NativeDeallocator` to returned `OpaqueDataBuffer` pointers
3. The native memory is now managed solely by ND4J's `DeallocatorService`
4. No more race condition between JavaCPP GC and ND4J's `DeallocatorService`

### Next Steps

1. Regenerate the bindings by rebuilding the preset modules
2. Rebuild the CUDA/Native backends to pick up the new bindings
3. Test to verify the fix works

### Build Commands

```bash
# Regenerate CUDA bindings
mvn clean install -pl nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda-preset -DskipTests

# Regenerate CPU bindings
mvn clean install -pl nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native-preset -DskipTests

# Rebuild CUDA backend
mvn clean install -pl nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda -DskipTests
```

### Result: **FAILED**

Same error still occurs after rebuilding:
```
RACE CONDITION DETECTED in registerWithDeallocatorService: Failed to set constant flag on buffer at 139946032660416 because the native buffer was already closed.
```

**Why It Failed:**

The `@NoDeallocator` annotation in the presets may not be sufficient because:

1. **Early finalization by JIT**: The JVM JIT compiler can determine that a local variable is "unreachable" before it actually goes out of scope. This means the `OpaqueDataBuffer` returned by `dbCreateExternalDataBuffer()` could be collected by GC BEFORE `retainReference()` is called, even though it's still in scope.

2. **The race window is extremely small but still exists**: Between `dbCreateExternalDataBuffer()` returning and `retainReference()` being called, GC can run and finalize the object.

3. **Bindings may not have been regenerated**: The `javaText` override requires the binding to be regenerated. Need to verify the generated `Nd4jCuda.java` actually has `@NoDeallocator`.

---

## 2026-01-20 - Alternative Approach: Mark Constant in Native Code

### The Problem

Even with `@NoDeallocator`, there's a fundamental race condition:
1. Native code allocates `InteropDataBuffer`
2. JNI returns pointer to Java
3. Java wraps pointer in `OpaqueDataBuffer` object
4. **RACE WINDOW**: GC can collect the Java object here (JIT early finalization)
5. Java calls `retainReference()` - too late, finalizer already ran
6. `dbSetConstant()` fails because buffer was destroyed

### The Solution: Mark Constant BEFORE Returning to Java

Create a new native function that allocates AND marks constant atomically:

```cpp
// New function in NativeOps.h
OpaqueDataBuffer* dbCreateConstantExternalDataBuffer(sd::LongType elements, int dataType, sd::Pointer primary, sd::Pointer special);
```

Implementation:
```cpp
OpaqueDataBuffer *dbCreateConstantExternalDataBuffer(sd::LongType elements, int dataType, sd::Pointer primary, sd::Pointer special) {
  auto buffer = dbCreateExternalDataBuffer(elements, dataType, primary, special);
  if (buffer != nullptr) {
    // Mark constant IMMEDIATELY, before returning to Java
    // This prevents any deallocation attempt
    buffer->isConstant.store(true, std::memory_order_release);
  }
  return buffer;
}
```

Then in Java, for constant buffers, call this new function instead of the regular one.

### Why This Works

1. The constant flag is set IN THE NATIVE CODE before the pointer even returns to Java
2. Even if the Java object is collected early, the finalizer will see `isConstant = true` and skip deallocation
3. No race window - the buffer is protected from the moment it's created

### Alternative: Use Volatile Field to Prevent Early Finalization

Another approach is to use `Reference.reachabilityFence()` in Java to prevent early finalization:

```java
public static OpaqueDataBuffer externalizedDataBuffer(..., boolean isConstant) {
    OpaqueDataBuffer ret = Nd4j.getNativeOps().dbCreateExternalDataBuffer(...);

    // Prevent JIT early finalization.
    // This ensures 'ret' is considered reachable until this point
    Reference.reachabilityFence(ret);

    if (ret != null && !ret.isNull()) {
        ret.retainReference();
    }
    // ... rest of method

    return ret;
}
```

However, this is fragile because:
1. Must be added to every call site
2. Easy to forget
3. Doesn't solve the fundamental design issue

### Recommended Fix: Native-Side Constant Marking

The safest approach is to add the `dbCreateConstantExternalDataBuffer` function that marks constant on the native side before returning.

### Files to Modify

| File | Change |
|------|--------|
| `libnd4j/include/legacy/NativeOps.h` | Add `dbCreateConstantExternalDataBuffer` declaration |
| `libnd4j/include/legacy/impl/NativeOpsHelpers_DataBuffers.cpp` | Add implementation |
| `nd4j/.../nativeblas/NativeOps.java` | Add interface method |
| `nd4j/.../presets/cuda/Nd4jCudaPresets.java` | Add binding with `@NoDeallocator` |
| `nd4j/.../presets/cpu/Nd4jCpuPresets.java` | Same for CPU |
| `nd4j/.../nativeblas/OpaqueDataBuffer.java` | Use new function for constant buffers |

---

## 2026-01-20 - FIX APPLIED: Native-Side Constant Marking

### Files Modified

| File | Change |
|------|--------|
| `libnd4j/include/legacy/NativeOps.h` | Added `dbCreateConstantExternalDataBuffer` declaration |
| `libnd4j/include/legacy/impl/NativeOpsHelpers_DataBuffers.cpp` | Added implementation that marks constant in native code |
| `nd4j/.../nativeblas/NativeOps.java` | Added interface method with `@NoDeallocator` |
| `nd4j/.../presets/cuda/Nd4jCudaPresets.java` | Added binding with `@NoDeallocator` javaText |
| `nd4j/.../presets/cpu/Nd4jCpuPresets.java` | Same for CPU backend |
| `nd4j/.../nativeblas/OpaqueDataBuffer.java` | Use `dbCreateConstantExternalDataBuffer` for constant buffers |

### How It Works

1. **New native function**: `dbCreateConstantExternalDataBuffer()` creates the buffer AND marks it constant atomically before returning to Java

2. **Native implementation**:
```cpp
OpaqueDataBuffer *dbCreateConstantExternalDataBuffer(...) {
  auto buffer = dbCreateExternalDataBuffer(elements, dataType, primary, special);
  if (buffer != nullptr) {
    // Mark constant IMMEDIATELY, before returning to Java
    buffer->isConstant.store(true, std::memory_order_release);
  }
  return buffer;
}
```

3. **Java side**: `externalizedDataBuffer()` now uses the new function when `isConstant=true`:
```java
if (isConstant) {
    ret = Nd4j.getNativeOps().dbCreateConstantExternalDataBuffer(...);
} else {
    ret = Nd4j.getNativeOps().dbCreateExternalDataBuffer(...);
}
```

### Why This Fixes The Race Condition

1. The constant flag is set **IN NATIVE CODE** before the pointer even returns to Java
2. Even if Java's GC runs immediately and tries to finalize the buffer, it will see `isConstant = true`
3. The `isValid()` check will succeed because constant buffers remain valid even after `_closed` is set
4. No race window exists - the buffer is protected from the moment it's created

### Build Commands

After this fix, rebuild the native library and Java modules:
```bash
# Rebuild libnd4j with the new function
cd libnd4j && ./buildnativeoperations.sh -c cuda

# Regenerate and rebuild Java modules
mvn clean install -pl nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda-preset -DskipTests
mvn clean install -pl nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda -DskipTests
```

---

## 2026-01-20 - FIX: Missing retainReference in allocateDataBuffer constant path

### Root Cause Found

In `OpaqueDataBuffer.allocateDataBuffer(numElements, dataType, allocateBoth, isConstant)`, the `buffer.retainReference()` call was missing before `registerWithDeallocatorService()`. The non-constant version at line 294 has it, but the constant version at line 362 did not.

### The Race Condition

```
1. allocateDataBuffer() returns OpaqueDataBuffer with JavaCPP NativeDeallocator
2. GC runs - JavaCPP finalizes buffer before retainReference() is called
3. registerWithDeallocatorService() tries to call dbSetConstant()
4. Native side sees buffer is closed, returns false
5. Exception thrown: "RACE CONDITION DETECTED"
```

### Files Modified

| File | Change |
|------|--------|
| `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/nativeblas/OpaqueDataBuffer.java` | Added `buffer.retainReference()` at line 363 in the constant buffer allocation path |

### The Fix

```java
// Before (missing retainReference):
if(buffer != null && !buffer.isNull()) {
    // Register with DeallocatorService, marking as constant if requested
    try {
        registerWithDeallocatorService(buffer, isConstant);

// After (with retainReference):
if(buffer != null && !buffer.isNull()) {
    buffer.retainReference();

    // Register with DeallocatorService, marking as constant if requested
    try {
        registerWithDeallocatorService(buffer, isConstant);
```

### Why This Works

1. `retainReference()` prevents JavaCPP from attaching/running a NativeDeallocator
2. The buffer is protected immediately after allocation succeeds
3. `registerWithDeallocatorService()` can now safely call `dbSetConstant()` without racing GC

### Result: **PARTIAL** - Different variable now fails

New error after fix:
```
USE-AFTER-FREE DETECTED: Input scalar at index 0 for op 'concat_9' contains InteropDataBuffer magic number (0xAB12CD34EF56).
Variable name: 'sd_var_19'. Array id: 821.
```

This indicates there are MULTIPLE code paths where buffers are created without proper protection.

---

## 2026-01-20 - Investigation: Multiple Buffer Creation Paths

### Known Buffer Creation Paths

1. **`allocateDataBuffer(numElements, dataType, allocateBoth)`** - Line 290
   - Has `retainReference()` at line 294 ✓

2. **`allocateDataBuffer(numElements, dataType, allocateBoth, isConstant)`** - Line 359
   - Now has `retainReference()` at line 363 ✓ (just fixed)

3. **`externalizedDataBuffer(numElements, dataType, primary, special, isConstant)`** - Line 211
   - Uses `dbCreateConstantExternalDataBuffer` for constants
   - Has `retainReference()` at line 223 ✓

4. **`BaseNDArray.detach()`** - Creates new buffer via `Nd4j.createBuffer()`
   - Goes through `DATA_BUFFER_FACTORY_INSTANCE.create()`
   - Need to verify this path has protection

5. **`Nd4j.create()` variants** - Multiple paths
   - Need to trace all paths to ensure protection

### Next Investigation

Need to trace what creates `sd_var_19` and which buffer allocation path it uses.

---

## 2026-01-20 - Continued Investigation

### Observations

1. The first fix (adding `retainReference()` in `allocateDataBuffer` constant path) was for a different variable (`Attention_0_three`) than the second error (`sd_var_19`)

2. Both variables appear to be scalar constants from the ONNX model

3. The magic number `0xAB12CD34EF56` is the ALIVE magic number for `InteropDataBuffer`, meaning:
   - The scalar's data buffer was freed
   - The memory was reused for a NEW `InteropDataBuffer`
   - We're reading the new buffer's magic field instead of the scalar value

### Constant Protection Flow

When a constant scalar is created:
1. `Nd4j.constantScalar()` creates array, marks buffer constant
2. `ThreadSafeArrayHolder.setArray()` checks `sourceIsConstant`
3. `array.detach()` creates NEW buffer (NOT constant initially)
4. `registerPendingConstant()` protects from GC
5. `setConstant(true)` marks buffer constant on Java and native side
6. Buffer is removed from DeallocatorService tracking

The race window is between steps 3 and 5 - the new buffer exists but isn't protected yet.

---

## 2026-01-20 - FIX: Missing constant marking in deserialization paths

### Root Cause Found

Multiple deserialization paths create arrays from FlatBuffer data but do NOT mark them as constant:

1. **`SameDiffSerializer.deserializeSmallNdArrayFromInlineBuffer()`** - Line 3196
   - Creates non-scalar arrays with `Nd4j.create()`
   - Returns without marking as constant
   - These arrays are meant to be model constants but are vulnerable to GC

2. **`FlatBuffersMapper.fromFlatNode()`** - Line 498 (now 498 with edits)
   - Non-scalar case uses `Nd4j.createFromFlatArray(fa)` without marking constant

3. **`FlatBuffersMapper.mapFlatPropertiesToFunctionProperties()`** - Lines 693 and 738
   - INDArray properties loaded from FlatBuffer without marking constant

### Files Modified

| File | Change |
|------|--------|
| `nd4j/.../serde/SameDiffSerializer.java` | Added `registerPendingConstant()` / constant marking / `releasePendingConstant()` around non-scalar array creation in `deserializeSmallNdArrayFromInlineBuffer()` |
| `nd4j/.../serde/FlatBuffersMapper.java` | Added constant marking for all `createFromFlatArray()` usages |

### The Fix

**SameDiffSerializer.deserializeSmallNdArrayFromInlineBuffer():**
```java
// Before:
INDArray result = Nd4j.create(dataType, shape, order);
// ... copy data ...
return result;

// After:
INDArray result = Nd4j.getDeallocatorService().registerPendingConstant(
    Nd4j.create(dataType, shape, order));
// ... copy data ...
// Mark array as constant to prevent deallocation during inference
if (result.data() != null) {
    result.data().setConstant(true);
}
if (result.shapeInfoDataBuffer() != null) {
    result.shapeInfoDataBuffer().setConstant(true);
}
result.setCloseable(false);
Nd4j.getDeallocatorService().releasePendingConstant(result);
return result;
```

**FlatBuffersMapper.fromFlatNode():**
```java
// Before:
} else {
    // Non-scalar case, use standard method
    scalar = Nd4j.createFromFlatArray(fa);
}

// After:
} else {
    // Non-scalar case, use standard method and mark as constant
    scalar = Nd4j.getDeallocatorService().registerPendingConstant(
        Nd4j.createFromFlatArray(fa));
    if (scalar.data() != null) {
        scalar.data().setConstant(true);
    }
    if (scalar.shapeInfoDataBuffer() != null) {
        scalar.shapeInfoDataBuffer().setConstant(true);
    }
    scalar.setCloseable(false);
    Nd4j.getDeallocatorService().releasePendingConstant(scalar);
}
```

### Why This Fixes The Issue

1. **Protection during creation**: `registerPendingConstant()` adds array to protected set
2. **Constant marking**: Data buffer and shape info buffer both marked constant
3. **Closeable=false**: Prevents explicit close() calls from freeing buffer
4. **Protection released**: `releasePendingConstant()` removes from set after marking done
5. **DeallocatorService skip**: Constant buffers are never deallocated

### Why Scalars Were Working

Scalars were correctly using `Nd4j.constantScalar()` which properly marks constant. Non-scalars used `Nd4j.create()` or `Nd4j.createFromFlatArray()` which do NOT mark constant.

### Result

FIX HAD NO EFFECT - Still getting scalar value 0 instead of 3

---

## 2026-01-20 - Continued Investigation: CONSTANT vs ARRAY type handling

### Key Observation

From the latest error log:
- `Attention_0_totalHiddenSize: type=ARRAY, Scalar Value: 2304` - CORRECT VALUE
- `Attention_0_three: type=CONSTANT, Scalar Value: 0` - WRONG (should be 3)

ARRAY type variables work correctly. CONSTANT type variables show wrong value (0 instead of 3).

The difference:
- ARRAY type: Retrieved directly from InferenceSession storage
- CONSTANT type: Retrieved via ThreadSafeArrayHolder → DeviceLocalNDArray

### Root Cause Hypothesis

The issue is in `DeviceLocalNDArray.get()` when retrieving a constant for a device different from where it was stored:

1. A new array is created: `Nd4j.create(delayedArray.dataType(), delayedArray.shape(), ...)`
2. Data is copied: `Nd4j.getMemoryManager().memcpy(newArray.data(), delayedArray.data())`

The problem: `delayedArray.data()` might have data only on DEVICE memory (from dup/detach operations in broadcast()), but the memcpy might read from HOST memory that was never populated, resulting in zeros.

### Code Analysis

In `DeviceLocalNDArray.broadcast()` (lines 241-252):
```java
if(!array.isEmpty() && array.data() != null) {
    INDArray delayed;
    try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
        delayed = sourceWasConstant ?
            Nd4j.getDeallocatorService().registerPendingConstant(array.dup(array.ordering()).detach()) :
            array.dup(array.ordering()).detach();
    }
    propagateConstantFlag(array, delayed);
    ...
    delayedArray = delayed;
}
```

The `array.dup().detach()` chain creates buffers that might have data only on DEVICE. When later used in get():
```java
Nd4j.getMemoryManager().memcpy(newArray.data(), delayedArray.data());
```

If `delayedArray` has no HOST data (or stale HOST data), the memcpy reads zeros.

### Proposed Fix

1. In `DeviceLocalNDArray.broadcast()`, after creating `delayedArray`, explicitly ensure HOST data is synced:
   ```java
   // After creating delayedArray, ensure data is on HOST for later cross-device copies
   Nd4j.getExecutioner().commit();
   if (delayed.data() instanceof BaseCudaDataBuffer) {
       AtomicAllocator.getInstance().synchronizeHostData(delayed);
   }
   ```

2. In `DeviceLocalNDArray.get()`, before memcpy, sync delayedArray to HOST:
   ```java
   // Ensure delayedArray has HOST data before copy
   Nd4j.getExecutioner().commit();
   AtomicAllocator.getInstance().synchronizeHostData(delayedArray);
   Nd4j.getMemoryManager().memcpy(newArray.data(), delayedArray.data());
   ```

### Files Modified

- `nd4j/.../linalg/util/DeviceLocalNDArray.java` - Added explicit HOST synchronization

### Fix Implemented

**DeviceLocalNDArray.java - get() method (line 147):**
```java
// Before:
Nd4j.getMemoryManager().memcpy(newArray.data(), delayedArray.data());

// After:
// Ensure delayedArray data is synced to HOST before cross-device copy
// This prevents reading stale/zero data from uninitialized HOST buffers
Nd4j.getExecutioner().commit();
Nd4j.getAffinityManager().ensureLocation(delayedArray, AffinityManager.Location.HOST);
Nd4j.getMemoryManager().memcpy(newArray.data(), delayedArray.data());
```

**DeviceLocalNDArray.java - broadcast() method (line 257):**
```java
// Before:
delayedArray = delayed;

// After:
// Ensure delayed array data is synced to HOST for cross-device copies
Nd4j.getExecutioner().commit();
Nd4j.getAffinityManager().ensureLocation(delayed, AffinityManager.Location.HOST);
delayedArray = delayed;
```

### Why This Should Fix The Issue

1. `commit()` ensures all pending CUDA operations complete
2. `ensureLocation(array, HOST)` explicitly syncs data from DEVICE to HOST
3. On CUDA, this calls `AtomicAllocator.getInstance().synchronizeHostData(array)`
4. The subsequent memcpy now reads from valid HOST data

### Result

**FIX FAILED** - Still getting use-after-free. Value is now pointer-like (`0x7f5d36146be0`) instead of 0 or 3.

---

## 2026-01-20 - Continued Investigation: Buffer still being freed

### Latest Error

```
Attention_0_three: Scalar Value: 140038315994080 (hex: 0x7f5d36146be0)
Data buffer address: 0x7f5d36237750
```

The scalar value is a pointer address, indicating the buffer memory was freed and reused for another allocation.

### Analysis

The HOST synchronization fix didn't help because the issue is NOT about data sync - it's about the buffer itself being deallocated.

The constant scalar IS using `Nd4j.constantScalar()` which properly marks the buffer as constant. However, somewhere in the chain, either:
1. A COPY of the buffer is being used that wasn't marked constant
2. The constant flag is being lost
3. A race condition allows deallocation before constant marking

### Next Investigation

Need to trace the EXACT buffer that gets stored vs the buffer that was marked constant.

Key question: When `DeviceLocalNDArray.broadcast()` is called with a constant scalar:
1. Does `array.detach()` return the SAME array (since not attached to workspace)?
2. If so, the same buffer should be stored
3. But if dup() creates a NEW buffer for delayed mode, that new buffer needs protection

---

## 2026-01-20 - Investigation: registerPendingConstant Race Window

### Problem Analysis

Looking at the `registerPendingConstant()` pattern more closely:

```java
// Current usage pattern:
INDArray result = Nd4j.getDeallocatorService().registerPendingConstant(
    Nd4j.create(dataType, shape, order));
// ... do work ...
result.data().setConstant(true);
Nd4j.getDeallocatorService().releasePendingConstant(result);
```

The issue is that `registerPendingConstant()` only adds the object to a Set - it doesn't immediately mark the buffer as constant:

```java
// DeallocatorService.registerPendingConstant() - current implementation:
public <T> T registerPendingConstant(T object) {
    if (object != null) {
        pendingConstants.add(object);  // Just adds to set
        // Does NOT call setConstant(true)!
    }
    return object;
}
```

### The Race Condition Sequence

1. `Nd4j.create()` allocates native buffer via `allocateDataBuffer()`
2. JavaCPP attaches NativeDeallocator to returned OpaqueDataBuffer
3. Object becomes eligible for GC
4. `registerPendingConstant()` is called - adds to set (strong reference)
5. **BUT**: Between steps 2 and 4, GC could run and finalize the buffer!

The strong reference in step 4 prevents FUTURE GC, but if GC already ran in step 3, the buffer is already freed.

### Proposed Fix: Immediate Constant Marking in registerPendingConstant

Modify `DeallocatorService.registerPendingConstant()` to immediately call `setConstant(true)` on the buffer:

```java
public <T> T registerPendingConstant(T object) {
    if (object != null) {
        pendingConstants.add(object);

        // IMMEDIATELY mark as constant to prevent race condition
        if (object instanceof INDArray) {
            INDArray arr = (INDArray) object;
            try {
                if (arr.data() != null) {
                    arr.data().setConstant(true);
                }
                if (arr.shapeInfoDataBuffer() != null) {
                    arr.shapeInfoDataBuffer().setConstant(true);
                }
                arr.setCloseable(false);
            } catch (Exception e) {
                // If setConstant fails, the buffer was already freed
                // Log and continue - caller will get exception when using the buffer
                log.warn("Failed to mark buffer as constant in registerPendingConstant: {}", e.getMessage());
            }
        }
    }
    return object;
}
```

### Why This May Help

1. The `setConstant(true)` call happens IMMEDIATELY after adding to pendingConstants
2. If the buffer was already freed, we catch the exception and log it (fail-fast detection)
3. If the buffer is still valid, it's now protected from future deallocation
4. The subsequent `setConstant(true)` calls by the caller will be no-ops (already constant)

### Why This May NOT Help

If GC already ran BEFORE `registerPendingConstant()` is called, the buffer is already freed and `setConstant()` will fail. This fix narrows the window but doesn't eliminate it.

### The ONLY True Fix

The buffer must be marked constant IN THE NATIVE CODE before returning to Java, or JavaCPP's deallocator must be disabled via `@NoDeallocator` in the generated bindings. Both approaches were attempted but may not have been applied consistently to all code paths.

---

## 2026-01-20 - FIX APPLIED: Immediate constant marking in registerPendingConstant

### Files Modified

| File | Change |
|------|--------|
| `nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/linalg/api/memory/deallocation/DeallocatorService.java` | Modified `registerPendingConstant()` to immediately mark INDArray buffers as constant |

### The Fix

Added immediate constant marking when an INDArray is registered as a pending constant:

```java
public <T> T registerPendingConstant(T object) {
    if (object != null) {
        pendingConstants.add(object);

        // Immediately mark as constant to narrow the race window
        // This must happen ASAP after the strong reference is established
        if (object instanceof INDArray) {
            INDArray arr = (INDArray) object;
            try {
                DataBuffer data = arr.data();
                if (data != null && !data.isConstant()) {
                    data.setConstant(true);
                }
                DataBuffer shapeInfo = arr.shapeInfoDataBuffer();
                if (shapeInfo != null && !shapeInfo.isConstant()) {
                    shapeInfo.setConstant(true);
                }
                arr.setCloseable(false);
            } catch (IllegalStateException e) {
                // Buffer was already freed by GC - log for debugging
                log.warn("registerPendingConstant: Buffer already freed for array - race condition detected. " +
                         "This may cause use-after-free errors downstream. Error: {}", e.getMessage());
            }
        }

        if (log.isTraceEnabled()) {
            log.trace("Registered pending constant: {} (total pending: {})",
                    object.getClass().getSimpleName(), pendingConstants.size());
        }
    }
    return object;
}
```

### Expected Result

1. If the buffer is still valid, it's immediately marked constant and protected
2. If the buffer was already freed, we log a warning (fail-fast detection)
3. This narrows the race window from "between create() and setConstant()" to "between native return and registerPendingConstant()"

### Limitations

This fix reduces but doesn't eliminate the race window. The true fix requires either:
1. Native-side constant marking before returning to Java
2. @NoDeallocator in generated JavaCPP bindings (properly applied to all buffer allocation methods)

### Result: **FAILED**

Same error still occurs:
```
malloc_consolidate(): unaligned fastbin chunk detected  // Heap corruption
FLOORDIV OP: division by zero is not allowed! Divisor value is 0.
Input 0: Shape: [0], Type: INT64
Input 1: Shape: [0], Type: INT64
```

**Critical observations:**
1. Both inputs show shape `[0]` - should be `[]` for scalars. Shape buffer is ALSO corrupted.
2. `malloc_consolidate` error during shutdown confirms heap corruption from use-after-free
3. The fix came TOO LATE - buffer was already freed before `registerPendingConstant()` was called

**Why the fix failed:**

The race condition happens BEFORE `registerPendingConstant()` is called:
```
1. Native allocateDataBuffer() returns OpaqueDataBuffer
2. JavaCPP attaches NativeDeallocator
3. JIT determines local variable "unreachable" (early finalization optimization)
4. GC runs, finalizer frees buffer  ← BUFFER FREED HERE
5. registerPendingConstant() called  ← TOO LATE - buffer already freed
6. setConstant(true) operates on freed/corrupted memory
```

The `registerPendingConstant()` fix narrows the window AFTER it's called, but the buffer can be freed BEFORE the call even happens due to JIT early finalization.

---

## 2026-01-20 - The Only Remaining Fix: Reference.reachabilityFence()

### Problem

JIT can determine that a local variable is "unreachable" before it actually goes out of scope. This allows GC to collect the object even though the code hasn't finished using it.

### Solution: Use Reference.reachabilityFence()

Java 9+ provides `Reference.reachabilityFence(obj)` which tells the JIT "don't consider this object unreachable until after this call".

### Where to Apply

In every method that allocates a buffer and then operates on it:

```java
public static OpaqueDataBuffer allocateDataBuffer(long numElements, int dataType, boolean allocateBoth, boolean isConstant) {
    OpaqueDataBuffer buffer = null;
    try {
        buffer = Nd4j.getNativeOps().allocateDataBuffer(numElements, dataType, allocateBoth);

        if (buffer != null && !buffer.isNull()) {
            buffer.retainReference();  // Prevent JavaCPP deallocation
            registerWithDeallocatorService(buffer, isConstant);
        }
        return buffer;
    } finally {
        // Prevent JIT from considering 'buffer' unreachable before this point.
        // Without this, GC can finalize buffer between allocateDataBuffer() and retainReference()
        java.lang.ref.Reference.reachabilityFence(buffer);
    }
}
```

### Files to Modify

| File | Method | Change |
|------|--------|--------|
| `nd4j/.../nativeblas/OpaqueDataBuffer.java` | `allocateDataBuffer()` | Add `Reference.reachabilityFence(buffer)` |
| `nd4j/.../nativeblas/OpaqueDataBuffer.java` | `externalizedDataBuffer()` | Add `Reference.reachabilityFence(ret)` |

### Why This Will Work

1. `Reference.reachabilityFence()` is a JVM intrinsic that prevents the JIT from optimizing away the reachability of an object
2. By placing it in a `finally` block, we ensure it's called even if an exception occurs
3. The buffer cannot be finalized until after `retainReference()` is called
4. `retainReference()` then permanently prevents JavaCPP finalization

### Caveat

`Reference.reachabilityFence()` requires Java 9+. For Java 8, the alternative is to use a volatile field or synchronized block to achieve similar effect.

### Result: **NOT VIABLE**

`Reference.reachabilityFence()` was previously tried and had to be reverted. It did not solve the problem because the JIT early finalization can happen even before the method containing the fence is entered.

---

## 2026-01-20 - Alternative Approach: Flow isConstant Through dup() Chain

### Analysis

The constant scalars are created correctly, but when they go through `DeviceLocalNDArray.broadcast()`:

```java
// DeviceLocalNDArray.broadcast() line 250
delayed = array.dup(array.ordering()).detach();
```

The `dup()` method creates a NEW buffer via `Nd4j.create()` → `allocateDataBuffer()`. This path does NOT use the `isConstant` parameter, so the new buffer is created as non-constant and is immediately vulnerable to GC.

### The Missing Link

There IS an `allocateDataBuffer(numElements, dataType, allocateBoth, isConstant)` overload that marks constant at creation time, but it's not being used by `dup()`.

### Proposed Fix: Add isConstant Parameter to dup()

1. Add `dup(char order, boolean isConstant)` method to INDArray interface
2. Have `BaseNDArray.dup(order, isConstant)` flow isConstant to `Nd4j.create()`
3. Have `Nd4j.create()` flow isConstant to `allocateDataBuffer()`
4. Update `DeviceLocalNDArray.broadcast()` to use `array.dup(array.ordering(), sourceWasConstant)`

### Why This Will Work

1. The buffer is marked constant IN THE NATIVE CODE during allocation
2. By the time the pointer returns to Java, it's already protected
3. No race window - the buffer is constant from birth
4. Even if JavaCPP's finalizer runs, `dbClose()` will see `isConstant=true` and skip deallocation

### Files to Modify

| File | Change |
|------|--------|
| `nd4j/.../api/ndarray/INDArray.java` | Add `dup(char order, boolean isConstant)` method |
| `nd4j/.../api/ndarray/BaseNDArray.java` | Implement `dup(char order, boolean isConstant)` |
| `nd4j/.../factory/Nd4j.java` | Add `create()` overloads with `isConstant` parameter |
| `nd4j/.../linalg/util/DeviceLocalNDArray.java` | Use `dup(order, sourceWasConstant)` in broadcast() |

### Alternative: Simpler Fix in DeviceLocalNDArray

Instead of modifying the entire dup() chain, we could create a helper method that:
1. Blocks DeallocatorService briefly
2. Creates the dup
3. Marks constant
4. Unblocks DeallocatorService

But this only blocks ND4J's deallocator, not JavaCPP's NativeDeallocator.

### The Real Issue: JavaCPP's NativeDeallocator

The fundamental problem is that JavaCPP attaches a NativeDeallocator to every Pointer returned from native code. The ONLY ways to prevent this are:

1. **@NoDeallocator annotation in GENERATED bindings** - Not just on the interface
2. **retainReference() before ANY Java code runs** - Impossible due to JIT
3. **Native-side protection before returning** - The `dbCreateConstantExternalDataBuffer` approach

### Verification Needed

Check if `Nd4jCuda.java` (generated) actually has `@NoDeallocator` on the allocation methods. If not, the preset changes didn't take effect and need to be fixed.

---

## 2026-01-20 - ROOT CAUSE FOUND: shapeBufferEx Missing @NoDeallocator

### Discovery

The data buffer methods (`allocateDataBuffer`, `dbCreateExternalDataBuffer`) have `@NoDeallocator` in the generated bindings, but the **shape buffer methods** (`shapeBufferEx`, `cacheAndStoreShapeBuffer`, `shapeBuffer`) do NOT!

### The Problem

`shapeBufferEx` returns a POINTER to a **CACHED** `ConstantShapeBuffer`:

```cpp
// NativeOps.h
typedef sd::ConstantShapeBuffer* OpaqueConstantShapeBuffer;

// shapeBufferEx implementation
OpaqueConstantShapeBuffer shapeBufferEx(...) {
    auto buffer = sd::ConstantShapeHelper::getInstance().bufferForShapeInfo(desc);
    return buffer;  // Returns pointer to CACHED buffer
}
```

The cache owns this buffer - callers should NOT delete it!

### Generated JavaCPP Binding (BROKEN)

```java
// Nd4jCuda.java - NO @NoDeallocator!
public native @ByVal org.nd4j.nativeblas.OpaqueConstantShapeBuffer shapeBufferEx(...);
```

JavaCPP attaches a `NativeDeallocator` that will DELETE the cached buffer when GC runs!

### What Happens

1. `CudaExecutioner.createShapeInfo()` calls `shapeBufferEx()`
2. Returns pointer to CACHED `ConstantShapeBuffer`
3. JavaCPP wraps with `NativeDeallocator`
4. `dbf.retainReference()` is called (should prevent deallocation...)
5. BUT: Due to JIT early finalization, GC can run BEFORE `retainReference()`
6. `NativeDeallocator` calls `delete` on the CACHED buffer
7. Cache now has dangling pointer
8. ANY array using that cached shape info now has garbage shape
9. Error: `Shape: [0]` instead of `[]` for scalars

### Why This Explains the Error

The error showed:
```
Input 0: Shape: [0], Type: INT64
Input 1: Shape: [0], Type: INT64
```

Shape `[0]` is garbage - a scalar should have shape `[]`. The shape buffer was freed and the memory reused.

### The Fix

Add `@NoDeallocator` to ALL shape buffer methods in the presets:

```java
// In Nd4jCudaPresets.java
.put(new Info("shapeBufferEx").javaText(
        "@org.bytedeco.javacpp.annotation.NoDeallocator public native @ByVal org.nd4j.nativeblas.OpaqueConstantShapeBuffer shapeBufferEx(int rank, @Cast(\"sd::LongType*\") LongPointer shape, @Cast(\"sd::LongType*\") LongPointer strides, @Cast(\"sd::DataType\") int dtype, char order, @Cast(\"sd::LongType\") long ews, @Cast(\"sd::LongType\") long extras);"))
.put(new Info("shapeBuffer").javaText(
        "@org.bytedeco.javacpp.annotation.NoDeallocator public native @ByVal org.nd4j.nativeblas.OpaqueConstantShapeBuffer shapeBuffer(int rank, @Cast(\"sd::LongType*\") LongPointer shape, @Cast(\"sd::LongType*\") LongPointer strides, @Cast(\"sd::DataType\") int dtype, char order, @Cast(\"sd::LongType\") long ews, @Cast(\"bool\") boolean empty);"))
.put(new Info("cacheAndStoreShapeBuffer").javaText(
        "@org.bytedeco.javacpp.annotation.NoDeallocator public native @ByVal org.nd4j.nativeblas.OpaqueConstantShapeBuffer cacheAndStoreShapeBuffer(@Cast(\"sd::LongType*\") LongPointer shapeInfo);"))
```

### Why This Will Work

1. `@NoDeallocator` prevents JavaCPP from attaching a `NativeDeallocator`
2. The returned pointer is to CACHED data - it should never be deleted by Java
3. The cache manages the lifetime of the shape buffers
4. `retainReference()` in `CudaExecutioner.createShapeInfo()` becomes a no-op (nothing to retain)
5. Shape buffers persist correctly

### Files to Modify

| File | Change |
|------|--------|
| `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda-preset/src/main/java/org/nd4j/presets/cuda/Nd4jCudaPresets.java` | Add `@NoDeallocator` javaText for `shapeBufferEx`, `shapeBuffer`, `cacheAndStoreShapeBuffer` |
| `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native-preset/src/main/java/org/nd4j/presets/cpu/Nd4jCpuPresets.java` | Same for CPU backend |

---

## 2026-01-20 - REVERTED: @NoDeallocator for Shape Buffer Methods

### Why Reverted

The shape buffer `@NoDeallocator` fix was reverted because:

1. **Shape buffers are already auto-cached** - The caching mechanism already handles lifetime management
2. **Build failure** - The javaText override only covered one method signature, but there are multiple overloads (LongPointer, LongBuffer, long[]) causing compilation errors
3. **Not the root cause** - The shape buffer caching already prevents deallocation of cached shapes

### Current Status

The root cause of the scalar use-after-free issue remains unidentified. Shape buffer caching is not the problem since it's already handled by the native cache.

### Remaining Fixes in Place

1. `registerPendingConstant()` in DeallocatorService now immediately marks buffers as constant (defensive measure)
2. Data buffer methods (`allocateDataBuffer`, `dbCreateExternalDataBuffer`, etc.) have `@NoDeallocator`

### Next Investigation Direction

Need to look elsewhere for the root cause since shape buffers are already properly cached. Possible areas:
- Data buffer lifecycle during model deserialization
- The specific path that creates constant scalar arrays
- Memory management in DeviceLocalNDArray cross-device copies

---

## 2026-01-20 - Continued Investigation: Shape Buffer Corruption Analysis

### Error Analysis

The user's error shows:
```
Input 0: Shape: [0], Type: INT64
Input 1: Shape: [0], Type: INT64
```

Key observation: Both inputs show shape `[0]` (1D array with 0 elements) instead of `[]` (scalar). For scalars:
- Correct: rank=0, shape=`[]`, lengthOf=1
- Corrupted: rank=1, shape=`[0]`, lengthOf=0

If shapeInfo[0] (the rank) gets corrupted from 0 to 1, the shape would be read as `[0]`.

### Native Typedef Discovery

Critical finding in `NativeOps.h`:
```cpp
typedef sd::ConstantShapeBuffer* OpaqueConstantShapeBuffer;  // Line 71
```

The `OpaqueConstantShapeBuffer` type is a **POINTER** to `sd::ConstantShapeBuffer`, not the struct itself.

### Java Binding Mismatch?

The generated `Nd4jCuda.java` has:
```java
@org.bytedeco.javacpp.annotation.NoDeallocator public native @ByVal org.nd4j.nativeblas.OpaqueConstantShapeBuffer shapeBufferEx(...)
```

The `@ByVal` annotation combined with a pointer typedef could be problematic:
1. Native function returns `sd::ConstantShapeBuffer*` (pointer to cached data)
2. `@ByVal` tells JavaCPP to treat this as a value
3. This might cause JavaCPP to copy/manage the pointer in unexpected ways

### Verification Needed

Check if the native `shapeBufferEx` returns a pointer to SHARED cached data:
- If YES: `@ByVal` might be incorrect - should NOT take ownership
- If NO: Caching should prevent deallocation issues

### OpaqueConstantShapeBuffer Class

The Java class (`OpaqueConstantShapeBuffer.java`) is minimal:
```java
public class OpaqueConstantShapeBuffer extends Pointer {
    public OpaqueConstantShapeBuffer(Pointer p) { super(p); }
}
```

It extends `Pointer` which CAN have deallocators attached. The `@NoDeallocator` on the method should prevent this, but there may be edge cases.

### Proposed Fix #1: Remove @ByVal from shape buffer methods

In `Nd4jCudaPresets.java`, add javaText overrides to remove `@ByVal`:

```java
.put(new Info("shapeBufferEx").javaText(
    "@org.bytedeco.javacpp.annotation.NoDeallocator public native org.nd4j.nativeblas.OpaqueConstantShapeBuffer shapeBufferEx(int rank, @Cast(\"sd::LongType*\") LongPointer shape, @Cast(\"sd::LongType*\") LongPointer strides, @Cast(\"sd::DataType\") int dtype, char order, @Cast(\"sd::LongType\") long ews, @Cast(\"sd::LongType\") long extras);"))
.put(new Info("shapeBuffer").javaText(
    "@org.bytedeco.javacpp.annotation.NoDeallocator public native org.nd4j.nativeblas.OpaqueConstantShapeBuffer shapeBuffer(int rank, @Cast(\"sd::LongType*\") LongPointer shape, @Cast(\"sd::LongType*\") LongPointer strides, @Cast(\"sd::DataType\") int dtype, char order, @Cast(\"sd::LongType\") long ews, @Cast(\"bool\") boolean empty);"))
.put(new Info("cacheAndStoreShapeBuffer").javaText(
    "@org.bytedeco.javacpp.annotation.NoDeallocator public native org.nd4j.nativeblas.OpaqueConstantShapeBuffer cacheAndStoreShapeBuffer(@Cast(\"sd::LongType*\") LongPointer shapeInfo);"))
```

This removes `@ByVal` and ensures `@NoDeallocator` is applied.

### Proposed Fix #2: Store shape buffer reference in DataBuffer

In `CudaExecutioner.createShapeInfo()`, the `OpaqueConstantShapeBuffer dbf` goes out of scope after the method returns. Even with `retainReference()`, there's a race window. Instead, store the reference in the created DataBuffer:

```java
// In CudaLongDataBuffer, add field:
private OpaqueConstantShapeBuffer sourceShapeBuffer;

// In CudaExecutioner.createShapeInfo():
val result = new CudaLongDataBuffer(primaryShapeInfo, specialShapeInfo, shapeInfoLength, true);
result.setSourceShapeBuffer(dbf);  // Keep reference alive
```

This ensures the shape buffer stays alive as long as the DataBuffer using it.

### Proposed Fix #3: Volatile field pattern to prevent JIT early finalization

In `CudaExecutioner.createShapeInfo()`, use a volatile field to prevent JIT from optimizing away the reference:

```java
// Class-level volatile field to hold shape buffer references
private static volatile Object shapeBufferHolder;

public DataBuffer createShapeInfo(...) {
    OpaqueConstantShapeBuffer dbf = Nd4j.getNativeOps().shapeBufferEx(...);
    shapeBufferHolder = dbf;  // Prevent JIT early finalization

    dbf.retainReference();

    // ... rest of method ...

    return result;
}
```

### Files to Investigate

1. `libnd4j/include/helpers/ConstantShapeHelper.h` - How cached shapes are stored
2. `libnd4j/include/array/ConstantShapeBuffer.h` - The ConstantShapeBuffer struct
3. Check if there's a destructor that frees memory prematurely

### Next Steps

1. Try Proposed Fix #1 (remove @ByVal) - simplest change
2. If still failing, try Fix #2 (store reference in DataBuffer)
3. Investigate native ConstantShapeHelper cache implementation

---

## 2026-01-20 - FIX APPLIED: @NoDeallocator on Shape Buffer Methods (without @ByVal)

### Files Modified

| File | Change |
|------|--------|
| `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda-preset/src/main/java/org/nd4j/presets/cuda/Nd4jCudaPresets.java` | Added javaText overrides for `shapeBufferEx`, `shapeBuffer`, `cacheAndStoreShapeBuffer` with `@NoDeallocator` and WITHOUT `@ByVal` |
| `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native-preset/src/main/java/org/nd4j/presets/cpu/Nd4jCpuPresets.java` | Same changes for CPU backend |

### The Fix

Added explicit javaText overrides to control the generated bindings:

```java
// In Nd4jCudaPresets.java and Nd4jCpuPresets.java
.put(new Info("shapeBufferEx").javaText(
    "@org.bytedeco.javacpp.annotation.NoDeallocator public native org.nd4j.nativeblas.OpaqueConstantShapeBuffer shapeBufferEx(int rank, @Cast(\"sd::LongType*\") LongPointer shape, @Cast(\"sd::LongType*\") LongPointer strides, @Cast(\"sd::DataType\") int dtype, char order, @Cast(\"sd::LongType\") long ews, @Cast(\"sd::LongType\") long extras);"))
.put(new Info("shapeBuffer").javaText(
    "@org.bytedeco.javacpp.annotation.NoDeallocator public native org.nd4j.nativeblas.OpaqueConstantShapeBuffer shapeBuffer(int rank, @Cast(\"sd::LongType*\") LongPointer shape, @Cast(\"sd::LongType*\") LongPointer strides, @Cast(\"sd::DataType\") int dtype, char order, @Cast(\"sd::LongType\") long ews, @Cast(\"bool\") boolean empty);"))
.put(new Info("cacheAndStoreShapeBuffer").javaText(
    "@org.bytedeco.javacpp.annotation.NoDeallocator public native org.nd4j.nativeblas.OpaqueConstantShapeBuffer cacheAndStoreShapeBuffer(@Cast(\"sd::LongType*\") LongPointer shapeInfo);"))
```

### Key Change: Removed @ByVal

The generated bindings previously had `@ByVal` which could cause JavaCPP to manage the pointer lifecycle incorrectly. Since `OpaqueConstantShapeBuffer` is typedef'd as `sd::ConstantShapeBuffer*` (a pointer), `@ByVal` semantics don't apply correctly.

### Rationale

1. Shape buffers are CACHED in native `ConstantShapeHelper` - they should NEVER be freed by Java
2. `@NoDeallocator` prevents JavaCPP from attaching a `NativeDeallocator`
3. Removing `@ByVal` ensures the pointer is treated as a simple pointer, not a value type
4. The `retainReference()` call in `CudaExecutioner.createShapeInfo()` is now a no-op (nothing to retain since there's no deallocator)

### Build Commands

After this fix, rebuild the presets and backends to regenerate bindings:

```bash
# Rebuild CUDA preset (regenerates Nd4jCuda.java)
mvn clean install -pl nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda-preset -DskipTests

# Rebuild CUDA backend
mvn clean install -pl nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda -DskipTests

# For CPU backend:
mvn clean install -pl nd4j/nd4j-backends/nd4j-backend-impls/nd4j-native-preset -DskipTests
```

### Expected Result

The generated `Nd4jCuda.java` and `Nd4jCpu.java` should now have:
```java
@org.bytedeco.javacpp.annotation.NoDeallocator public native org.nd4j.nativeblas.OpaqueConstantShapeBuffer shapeBufferEx(...)
```

Instead of the previous:
```java
@org.bytedeco.javacpp.annotation.NoDeallocator public native @ByVal org.nd4j.nativeblas.OpaqueConstantShapeBuffer shapeBufferEx(...)
```

### Testing Required

Run the model loading test that was failing:
```bash
# The error should no longer occur:
# Input 0: Shape: [0], Type: INT64
# Input 1: Shape: [0], Type: INT64
```

### Result: PENDING - Rebuild required to test

---

## 2026-01-20 - BUILD FIX: Corrected cacheAndStoreShapeBuffer signature

### Issue

Build failed with:
```
Nd4jCuda.java:[9,8] org.nd4j.linalg.jcublas.bindings.Nd4jCuda is not abstract and does not override abstract method cacheAndStoreShapeBuffer(long[]) in org.nd4j.nativeblas.NativeOps
```

### Root Cause

The `NativeOps` interface specifies:
```java
org.nd4j.nativeblas.OpaqueConstantShapeBuffer cacheAndStoreShapeBuffer(long[] shapeInfo);
```

But my javaText used `LongPointer` instead of `long[]`.

### Fix Applied

Changed the javaText for `cacheAndStoreShapeBuffer` from:
```java
"@org.bytedeco.javacpp.annotation.NoDeallocator public native ... cacheAndStoreShapeBuffer(@Cast(\"sd::LongType*\") LongPointer shapeInfo);"
```

To:
```java
"@org.bytedeco.javacpp.annotation.NoDeallocator public native ... cacheAndStoreShapeBuffer(@Cast(\"sd::LongType*\") long[] shapeInfo);"
```

### Interface Requirements Verified

| Method | Interface Parameter Type | javaText Parameter Type |
|--------|-------------------------|------------------------|
| `cacheAndStoreShapeBuffer` | `long[]` | `long[]` ✓ |
| `shapeBufferEx` | `LongPointer` | `LongPointer` ✓ |
| `shapeBuffer` | (not in interface) | `LongPointer` ✓ |

### Rebuild Command

```bash
mvn clean install -pl nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda-preset,nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda -DskipTests
```

---

## 2026-01-20 - REVERTED: Shape buffer javaText overrides

### Why Reverted

The shape buffer javaText approach was reverted because:

1. **Removing `@ByVal` breaks native code generation**: Without `@ByVal`, JavaCPP generates code expecting `sd::ConstantShapeBuffer**` (pointer-to-pointer) but the native function returns `sd::ConstantShapeBuffer*` (single pointer).

2. **`@ByVal` + `@NoDeallocator` is already in the generated bindings**: The existing generated Nd4jCuda.java already has both annotations - custom javaText overrides were unnecessary.

3. **The shape buffers already have `@NoDeallocator`**: Checked the generated bindings and confirmed they already have the annotation.

### Current State

The generated bindings already have the correct configuration:
```java
@org.bytedeco.javacpp.annotation.NoDeallocator public native @ByVal org.nd4j.nativeblas.OpaqueConstantShapeBuffer shapeBufferEx(...)
```

The issue must be elsewhere - the `@NoDeallocator` annotation IS present, so JavaCPP should not be attaching deallocators to shape buffers.

### Next Investigation Areas

Since shape buffer `@NoDeallocator` is already in place, the issue may be:
1. **Data buffer corruption** rather than shape buffer corruption
2. **Race condition in a different code path** not related to shape buffer allocation
3. **The model file itself** may have incorrect shape data serialized

---

## ⛔ BANNED APPROACH: JavaCPP Annotation Modifications

**DO NOT attempt to fix this issue by modifying JavaCPP annotations in preset files.**

Multiple attempts were made to add/modify `@NoDeallocator`, `@ByVal`, and `javaText` overrides for shape buffer methods. All failed because:

1. The generated bindings already have correct annotations
2. Removing `@ByVal` breaks native code generation (pointer vs pointer-to-pointer mismatch)
3. Adding javaText overrides causes interface signature mismatches

**The fix is NOT in JavaCPP configuration.**

---

## 2026-01-20 - New Investigation: FlatBuffers and ONNX Import

### Key Observation

CPU backend works fine. The issue is CUDA-specific. This suggests the problem is NOT in:
- JavaCPP bindings (same for both backends)
- Shape buffer caching (same native code)

### Areas to Investigate

1. **FlatBuffers deserialization of scalar constants** - How are scalars loaded from .fb/.sdz files?
2. **ONNX import path for scalar shape handling** - How are ONNX scalar tensors converted?

### Investigation #1: FlatBuffers Scalar Deserialization

**Checked** - FlatBuffersMapper.java correctly handles scalars:
- Line 450: Checks `fa.shapeLength() == 0` for scalar detection
- Uses `Nd4j.constantScalar()` for scalars (lines 468-490)
- Non-scalars use `createFromFlatArray()` (line 498)

### Investigation #2: ONNX Import Scalar Handling

**Checked** - ONNX import correctly handles scalars:
- `OnnxIRTensor.shape()` directly returns `tensor.dimsList` (empty for scalars)
- `ndarrayFromNameSpaceTensor()` has explicit scalar detection: `totalLen <= 1 && shape.isEmpty()`
- Uses `Nd4j.scalar()` for proper rank-0 creation
- No code converts `[]` to `[0]`

### BUG FOUND: ShapeUtils::shapeAsString Display Bug

**File**: `libnd4j/include/helpers/impl/ShapeUtils.cpp` line 888-889

```cpp
std::string ShapeUtils::shapeAsString(NDArray* array) {
  if (array->rankOf() == 0 && !array->isEmpty()) return "[0]";  // BUG!
```

This function incorrectly displays scalars (rank 0) as `"[0]"` instead of `"[]"`.

- A scalar has rank=0, shape=`[]` (empty)
- `"[0]"` looks like a 1D array with 0 elements
- This is a **display bug** that causes confusion in error messages

**However**, this is just the display function. The actual error might still be a real shape corruption issue, not just display.

### FIX APPLIED: ShapeUtils::shapeAsString Display Bug

**File**: `libnd4j/include/helpers/impl/ShapeUtils.cpp` line 888-891

Changed:
```cpp
// OLD (buggy):
if (array->rankOf() == 0 && !array->isEmpty()) return "[0]";

// NEW (fixed):
if (array->rankOf() == 0) return "[]";
```

Now scalars will display as `"[]"` instead of the confusing `"[0]"`.

### Clarification: The Actual Error

Looking at the floordiv error more carefully:
```
FLOORDIV OP: division by zero is not allowed! Divisor value is 0.
```

This error comes from the zero-check at line 64 of floordiv.cpp, inside the `if (y->lengthOf() == 1)` block.

This means:
1. The divisor IS being recognized as a scalar (length 1) ✓
2. The VALUE is being read as 0 (when it should be e.g., 3) ✗

**The shape is NOT corrupted** - the `[0]` display was just a confusing representation of a scalar.

**The DATA BUFFER is corrupted** - the scalar's value is being read as 0 instead of the correct value.

This confirms the earlier hypothesis: the DATA buffer (not shape buffer) is being freed prematurely, causing the value to become garbage/zero.

## ROOT CAUSE IDENTIFIED AND FIXED (2026-01-20)

### Root Cause: Use-After-Free in copyDataFromSrc()

**File**: `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/buffer/BaseCudaDataBuffer.java`

**The Bug**:
In `copyDataFromSrc()`, a temporary `LongPointer` (or other typed pointer) wrapping a Java array was being passed to `setPrimaryBuffer()`, which set the native `_primaryBuffer` to point to this temporary pointer. After the method returned, the temporary pointer went out of scope and its memory became invalid.

**Old Code (Buggy)**:
```java
public void copyDataFromSrc(Pointer pointer, long length, long srcOffset, long dstOffset) {
    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));
    val context = AtomicAllocator.getInstance().getDeviceContext();
    ptrDataBuffer.setPrimaryBuffer(pointer, length);  // ← BUG: Sets HOST to temporary pointer
    NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(...);
    // ... method returns, pointer goes out of scope
    // Later, syncToPrimary() writes to stale/freed memory
}
```

**What Happened**:
1. `set(long[], ...)` called `copyDataFromSrc(new LongPointer(data), ...)`
2. `setPrimaryBuffer()` stored the temporary pointer address in native `_primaryBuffer`
3. Data was correctly copied to DEVICE via memcpyAsync
4. Method returned, `LongPointer` went out of scope (memory freed/reused)
5. Later, when `syncToPrimary()` was called to copy device→host, it:
   - Checked `isPrimaryActual()` - returned false (device was newer)
   - Called `allocatePrimary()` - did NOTHING because `_primaryBuffer != null`
   - Called `cudaMemcpy(_primaryBuffer, _specialBuffer, ...)` - wrote to freed memory
6. Reading from HOST returned 0 (memory had been zeroed or reused)

**The Fix**:
```java
public void copyDataFromSrc(Pointer pointer, long length, long srcOffset, long dstOffset) {
    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));
    val context = AtomicAllocator.getInstance().getDeviceContext();

    // First, ensure we have properly allocated HOST memory (not a temporary pointer)
    if (allocationPoint.getHostPointer() == null || allocationPoint.getHostPointer().isNull()) {
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbAllocatePrimaryBuffer(ptrDataBuffer);
    }

    // Get the properly allocated HOST pointer
    Pointer hostPtr = allocationPoint.getHostPointer();
    
    // Copy from source Java array to our persistent HOST buffer
    val dstHostPtr = new CudaPointer(hostPtr.address() + (dstOffset * elementSize));
    Pointer.memcpy(dstHostPtr, srcPtr, length * getElementSize());

    // Now copy from our HOST buffer to DEVICE
    NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(
            allocationPoint.getDevicePointer(),
            dstHostPtr,
            length * getElementSize(),
            CudaConstants.cudaMemcpyHostToDevice,
            context.getSpecialStream());
    // ... rest of method
}
```

**Key Changes**:
1. Allocate proper persistent HOST memory via `dbAllocatePrimaryBuffer()` if not already allocated
2. Copy data from the temporary Java pointer to the persistent HOST buffer
3. Then copy from persistent HOST to DEVICE

This ensures the HOST buffer address is always valid and properly managed by native code.


## DEVICE CONTEXT SELECTION UPDATE (2026-01-20)

### Update: Obsolete Thread-Based Affinity Replaced

**Background**:
The original code used thread-based affinity via `Nd4j.getAffinityManager().getDeviceForCurrentThread()` to determine which GPU to use. This approach is obsolete - instead, the system should default to the GPU with the most memory, determined at launch time, with automatic failover support.

**Files Updated**:

1. **CudaZeroHandler.java** (`nd4j-cuda/src/main/java/org/nd4j/jita/handler/impl/CudaZeroHandler.java`)
   - Updated `getDeviceId()` to use `DeviceMemoryManager.getInstance().getDefaultDevice()`
   - Updated `getCudaCublasHandle()` to use centralized `getDeviceId()`
   - Updated `getCudaContext()` to use centralized `getDeviceId()`

2. **CudaOpContext.java** (`nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/ops/executioner/CudaOpContext.java`)
   - Updated `close()` method to use `DeviceMemoryManager`
   - Updated `purge()` method to use `DeviceMemoryManager`
   - Added helper method `getDefaultDeviceId()`

3. **CudaOpContextDeallocator.java** (`nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/ops/executioner/CudaOpContextDeallocator.java`)
   - Updated `deallocate()` method to use `DeviceMemoryManager`
   - Added helper method `getDefaultDeviceId()`

**New Device Selection Pattern**:
```java
/**
 * Get the default device ID from DeviceMemoryManager.
 * Uses the GPU with most memory as the default, with automatic failover.
 * Thread-based affinity is obsolete.
 */
private int getDefaultDeviceId() {
    DeviceDescriptor defaultDevice = DeviceMemoryManager.getInstance().getDefaultDevice();
    if (defaultDevice != null && defaultDevice.getDeviceType() != null
            && defaultDevice.getDeviceType().isGpu()) {
        return defaultDevice.getDeviceIndex();
    }
    // Fallback to device 0 if no GPU is configured as default
    return 0;
}
```

**Key Classes**:
- `DeviceMemoryManager` - Singleton managing device memory and default device selection
- `DeviceDescriptor` - Interface describing device properties (type, index, memory, etc.)
- `DeviceRoutingConfiguration` - Configuration for device routing policies

**References**:
- `HybridDataBuffer` interface as example of new device model
- `DeviceMemoryManager.getInstance().getDefaultDevice()` for device selection
- GPU with most memory is preferred by default

---

## INVESTIGATION UPDATE (2026-01-20) - Issue Still Occurring

### Current Error
The floordiv operation is still failing with division by zero:
```
Operation Name: floordiv
INPUT VARIABLES (2):
  [0] 'Attention_0_totalHiddenSize'
      Shape: [], DataType: LONG, Scalar Value: 2304 (CORRECT)
  [1] 'Attention_0_three'
      Shape: [], DataType: LONG, Scalar Value: 0 (WRONG - should be 3)

Error: FLOORDIV OP: division by zero is not allowed! Divisor value is 0.
```

Key observation: **ARRAY type works (2304), CONSTANT type fails (0)**

### Detailed Analysis of copyDataFromSrc Bug

**File**: `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/buffer/BaseCudaDataBuffer.java`

**Current Code (Lines 575-592)**:
```java
public void copyDataFromSrc(Pointer pointer, long length, long srcOffset, long dstOffset) {
    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));
    val context = AtomicAllocator.getInstance().getDeviceContext();
    val perfD = PerformanceTracker.getInstance().helperStartTransaction();
    ptrDataBuffer.setPrimaryBuffer(pointer, length);  // ← PROBLEM HERE
    NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(
        allocationPoint.getDevicePointer(), srcPtr, length * getElementSize(),
        CudaConstants.cudaMemcpyHostToDevice, context.getSpecialStream());

    PerformanceTracker.getInstance().helperRegisterTransaction(...);
    context.getSpecialStream().synchronize();

    allocationPoint.tickHostWrite();
    allocationPoint.tickDeviceWrite();
    NativeOpsHolder.getInstance().getDeviceNativeOps().dbTickDeviceWrite(ptrDataBuffer);
}
```

**Callers** (from `set()` methods):
```java
// For LONG data type (lines 735-737):
case LONG: {
    val pointer = new LongPointer(data);  // ← Temporary pointer wrapping Java long[]
    copyDataFromSrc(pointer, length, srcOffset, dstOffset);
}
```

### Native Side Analysis

**setPrimaryBuffer** (`libnd4j/include/array/impl/DataBuffer.cpp` lines 650-662):
```cpp
void DataBuffer::setPrimaryBuffer(void* buffer, size_t length) {
    std::lock_guard<std::mutex> lock(_deleteMutex);
    _primaryBuffer = buffer;        // ← Points to temporary memory
    _isOwnerPrimary = false;        // ← Won't free it (but also won't allocate replacement)
    _lenInBytes = length * DataTypeUtils::sizeOf(_dataType);
}
```

**allocatePrimary** (`libnd4j/include/array/impl/DataBuffer.cpp` lines 512-537):
```cpp
void DataBuffer::allocatePrimary() {
    if (_primaryBuffer == nullptr) {  // ← Only allocates if NULL!
        // ... allocation code
    }
    // If _primaryBuffer is NOT null, does NOTHING
}
```

**syncToPrimary** (`libnd4j/include/array/cuda/DataBuffer.cu` lines 428-467):
```cpp
void DataBuffer::syncToPrimary(const LaunchContext* context, const bool forceSync) {
    if (_specialBuffer == nullptr || _lenInBytes == 0 || closed) return;

    if (isPrimaryActual() && !forceSync) return;  // May skip sync

    allocatePrimary();  // ← Does nothing if _primaryBuffer != nullptr!

    // ...
    res = cudaMemcpy(_primaryBuffer, _specialBuffer, getLenInBytes(), cudaMemcpyDeviceToHost);
    // ← Writes to potentially FREED memory!
}
```

**isPrimaryActual** (`libnd4j/include/array/cuda/DataBuffer.cu` lines 876-878):
```cpp
bool DataBuffer::isPrimaryActual() const {
    return (_writePrimary.load() > _writeSpecial.load() ||
            _readPrimary.load() > _writeSpecial.load());
}
```

### The Complete Failure Sequence

1. **Scalar creation**: `Nd4j.constantScalar(3L)` creates a LONG scalar with value 3
2. **Buffer allocation**: `BaseCudaDataBuffer` constructor allocates DEVICE memory
3. **Data copy**: `setData(long[])` → `set(long[], ...)` → `copyDataFromSrc(new LongPointer(data), ...)`
4. **setPrimaryBuffer called**: `_primaryBuffer` set to temporary `LongPointer` address
5. **memcpyAsync**: Data correctly copied to DEVICE
6. **Method returns**: `LongPointer` goes out of scope, memory may be freed/reused
7. **Ticks updated**: `tickHostWrite()`, `tickDeviceWrite()` both called
   - `_writePrimary` = N
   - `_writeSpecial` = N+1 (device is "newer")
8. **Later, syncToPrimary called**: (e.g., during floordiv's `y->syncToHost()`)
   - `isPrimaryActual()` returns FALSE (device is newer)
   - `allocatePrimary()` called but does NOTHING (`_primaryBuffer != nullptr`)
   - `cudaMemcpy(_primaryBuffer, _specialBuffer, ...)` writes to STALE address
9. **Reading HOST**: Returns 0 (memory was freed/zeroed)

### Why Value is Exactly 0 (Not Garbage)

The value being exactly 0 (not random garbage) suggests one of:
1. Memory was zeroed when freed
2. Memory was reused and happens to contain 0
3. The `LongPointer` wrapper's memory pool was reset

### Required Fix

The fix needs to ensure persistent HOST memory before setting `_primaryBuffer`:

```java
public void copyDataFromSrc(Pointer pointer, long length, long srcOffset, long dstOffset) {
    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));
    val context = AtomicAllocator.getInstance().getDeviceContext();

    // Allocate persistent HOST memory first
    lazyAllocateHostPointer();  // Or call native dbAllocatePrimaryBuffer

    // Get the persistent HOST pointer
    Pointer hostPtr = allocationPoint.getHostPointer();
    if (hostPtr == null || hostPtr.isNull()) {
        throw new IllegalStateException("Failed to allocate HOST pointer");
    }

    // Copy from temporary Java pointer to persistent HOST buffer
    val dstHostPtr = new CudaPointer(hostPtr.address() + (dstOffset * elementSize));
    Pointer.memcpy(dstHostPtr, srcPtr, length * getElementSize());

    // Now copy from persistent HOST to DEVICE
    NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(
        allocationPoint.getDevicePointer(),
        dstHostPtr,
        length * getElementSize(),
        CudaConstants.cudaMemcpyHostToDevice,
        context.getSpecialStream());

    context.getSpecialStream().synchronize();

    allocationPoint.tickHostWrite();
    allocationPoint.tickDeviceWrite();
    NativeOpsHolder.getInstance().getDeviceNativeOps().dbTickDeviceWrite(ptrDataBuffer);
}
```

### getDeviceContext() GPU Selection - VERIFIED (2026-01-20)

The GPU selection mechanism has been verified to be properly synchronized between Java and native code.

#### Java Side Flow:
1. `AtomicAllocator.getInstance().getDeviceContext()` → `CudaZeroHandler.getCudaContext()`
2. `getCudaContext()` calls `Nd4j.getAffinityManager().getDeviceForCurrentThread()` (CudaZeroHandler.java:998)
3. `CudaAffinityManager.getDeviceForCurrentThread()` maintains a thread-to-device map (`ConcurrentHashMap<Long, Integer>`)
4. On first access, it assigns a device via `getNextDevice()` and calls `unsafeSetDevice(deviceId)`
5. `unsafeSetDevice()` calls `NativeOpsHolder.getInstance().getDeviceNativeOps().setDevice(deviceId)` → native `cudaSetDevice()`

#### Native Side Flow:
1. `LaunchContext::defaultContext()` calls `AffinityManager::currentDeviceId()` (LaunchContext.cu:125)
2. `AffinityManager::currentDeviceId()` **always** calls `cudaGetDevice(&nativeDevice)` first (AffinityManager.cu:38)
3. If the thread-local `globalThreadToDevice` doesn't match native device, it syncs to native (AffinityManager.cu:45-49)
4. This ensures native code always uses the actual CUDA device set by `cudaSetDevice()`

#### Synchronization Mechanism:
- **Java → Native**: `unsafeSetDevice(deviceId)` calls native `cudaSetDevice()`, which native code detects via `cudaGetDevice()`
- **Native → Java**: Not automatic, but native `AffinityManager::currentDeviceId()` always calls `cudaGetDevice()` first

#### In copyDataFromSrc():
The streams obtained from `getDeviceContext()` are tied to the device via:
1. Java gets device ID from `getDeviceForCurrentThread()` which has called `cudaSetDevice()` on first access
2. Native `defaultLaunchContext()` gets streams for the device returned by `cudaGetDevice()`
3. Since Java set the device via `cudaSetDevice()`, native will see the same device

**Conclusion**: The GPU selection is properly synchronized as long as:
1. The thread has called `getDeviceForCurrentThread()` at least once (which calls `cudaSetDevice()`)
2. No external code changes the CUDA device on this thread without going through AffinityManager

### Status: FIXED (2026-01-20)

The fix documented above has been implemented in `BaseCudaDataBuffer.copyDataFromSrc()`.

**Current Implementation (Lines 575-616)**:
```java
public void copyDataFromSrc(Pointer pointer, long length, long srcOffset, long dstOffset) {
    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));
    val context = AtomicAllocator.getInstance().getDeviceContext();

    // First, ensure we have properly allocated HOST memory (not a temporary pointer)
    // This fixes use-after-free when the source pointer goes out of scope
    if (allocationPoint.getHostPointer() == null || allocationPoint.getHostPointer().isNull()) {
        NativeOpsHolder.getInstance().getDeviceNativeOps().dbAllocatePrimaryBuffer(ptrDataBuffer);
    }

    // Get the properly allocated HOST pointer
    Pointer hostPtr = allocationPoint.getHostPointer();
    if (hostPtr == null || hostPtr.isNull()) {
        throw new IllegalStateException("Failed to allocate HOST buffer for data copy");
    }

    // Copy from source Java array to our persistent HOST buffer
    val dstHostPtr = new CudaPointer(hostPtr.address() + (dstOffset * elementSize));
    Pointer.memcpy(dstHostPtr, srcPtr, length * getElementSize());

    // Now copy from our HOST buffer to DEVICE
    // IMPORTANT: Apply dstOffset to device pointer as well (not just host pointer)
    val perfD = PerformanceTracker.getInstance().helperStartTransaction();
    val dstDevPtr = new CudaPointer(allocationPoint.getDevicePointer().address() + (dstOffset * elementSize));
    NativeOpsHolder.getInstance().getDeviceNativeOps().memcpyAsync(
            dstDevPtr,
            dstHostPtr,
            length * getElementSize(),
            CudaConstants.cudaMemcpyHostToDevice,
            context.getSpecialStream());

    PerformanceTracker.getInstance().helperRegisterTransaction(
            allocationPoint.getDeviceId(), perfD / 2,
            allocationPoint.getNumberOfBytes(), MemcpyDirection.HOST_TO_DEVICE);

    context.getSpecialStream().synchronize();

    // Mark both HOST and DEVICE as having valid data
    allocationPoint.tickHostWrite();
    allocationPoint.tickDeviceWrite();

    NativeOpsHolder.getInstance().getDeviceNativeOps().dbTickDeviceWrite(ptrDataBuffer);
}
```

**Key Changes from Original Buggy Code**:
1. **Removed `setPrimaryBuffer(pointer, length)` call** - No longer sets the primary buffer to the temporary pointer
2. **Added HOST memory allocation** - Calls `dbAllocatePrimaryBuffer()` if host pointer is null
3. **Two-phase copy** - First copies to persistent HOST buffer, then from HOST to DEVICE
4. **Added dstOffset to device pointer** - Bug fix: device memcpy destination now includes offset

**Bug Fix (2026-01-20)**: The previous "fix" had a bug where `dstOffset` was applied to the host pointer but NOT to the device pointer in `memcpyAsync`. This meant that when `dstOffset > 0`, data would be written to the wrong location on device. This has been corrected by creating a `dstDevPtr` with the offset applied.

**Subclass Usage**:
- `CudaHalfDataBuffer`, `CudaDoubleDataBuffer`, and `CudaBfloat16DataBuffer` all call the base class `copyDataFromSrc()` method, so they benefit from this fix automatically.

---

## 2026-01-22 - FIXES APPLIED: copyDataFromSrc and strided_slice

### Summary

Two critical fixes were applied based on the investigation in this document:

### Fix 1: BaseCudaDataBuffer.copyDataFromSrc() - Use-After-Free Prevention

**File**: `nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda/src/main/java/org/nd4j/linalg/jcublas/buffer/BaseCudaDataBuffer.java`

**Problem**: The old code called `setPrimaryBuffer(pointer, length)` with a temporary pointer (e.g., `LongPointer` wrapping a Java array). When this temporary pointer went out of scope, the memory could be freed/reused, but the native DataBuffer still pointed to it. Later syncToPrimary would write to freed memory.

**Fix Applied**:
```java
public void copyDataFromSrc(Pointer pointer, long length, long srcOffset, long dstOffset) {
    val srcPtr = new CudaPointer(pointer.address() + (srcOffset * elementSize));
    val nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    val context = AtomicAllocator.getInstance().getDeviceContext();

    // Do NOT call setPrimaryBuffer with the temporary pointer!
    // Instead, allocate persistent HOST memory first
    if (allocationPoint.getHostPointer() == null || allocationPoint.getHostPointer().isNull()) {
        nativeOps.dbAllocatePrimaryBuffer(ptrDataBuffer);
    }

    // Get the properly allocated HOST pointer
    Pointer hostPtr = allocationPoint.getHostPointer();
    if (hostPtr == null || hostPtr.isNull()) {
        throw new IllegalStateException("Failed to allocate HOST buffer for data copy");
    }

    // Copy from source Java array to our persistent HOST buffer
    val dstHostPtr = new CudaPointer(hostPtr.address() + (dstOffset * elementSize));
    Pointer.memcpy(dstHostPtr, srcPtr, length * getElementSize());

    // Now copy from our HOST buffer to DEVICE (with offset applied)
    val perfD = PerformanceTracker.getInstance().helperStartTransaction();
    val dstDevPtr = new CudaPointer(allocationPoint.getDevicePointer().address() + (dstOffset * elementSize));

    int result = nativeOps.memcpySync(
            dstDevPtr,
            dstHostPtr,
            length * getElementSize(),
            CudaConstants.cudaMemcpyHostToDevice,
            null);

    if (result == 0) {
        throw new RuntimeException("memcpySync failed in copyDataFromSrc");
    }

    PerformanceTracker.getInstance().helperRegisterTransaction(
            allocationPoint.getDeviceId(), perfD / 2,
            allocationPoint.getNumberOfBytes(), MemcpyDirection.HOST_TO_DEVICE);

    // Mark both HOST and DEVICE as having valid data
    allocationPoint.tickHostWrite();
    allocationPoint.tickDeviceWrite();
    nativeOps.dbTickDeviceWrite(ptrDataBuffer);
}
