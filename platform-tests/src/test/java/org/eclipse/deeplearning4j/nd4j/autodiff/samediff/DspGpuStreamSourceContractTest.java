/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** Source guards only; runtime stream/capture behavior still requires the CUDA gates. */
@Tag("source-lint")
class DspGpuStreamSourceContractTest {
    private String source() throws Exception {
        return Files.readString(Path.of("..").toAbsolutePath().normalize().resolve(
                "libnd4j/include/graph/impl/NativeDynamicShapePlan_gpubackend.cu"));
    }

    @Test
    void streamIsNormalizedBeforeAnyGpuDispatch() throws Exception {
        String source = source();
        String entry = source.substring(source.indexOf(
                "Status NativeDynamicShapePlan::executeSegmentWithGpuGraph("));
        int secondary = entry.indexOf("stream = static_cast<void*>(&tl_secondarySegStream);");
        int resolve = entry.indexOf("cudaStream_t cudaStr = (stream != nullptr)");
        int normalize = entry.indexOf("stream = static_cast<void*>(&cudaStr);");
        assertTrue(secondary >= 0 && resolve > secondary && normalize > resolve);
        assertTrue(entry.substring(resolve, normalize).contains(
                "? *static_cast<cudaStream_t*>(stream) : nullptr;"), "Preserve explicit stream handles");
        assertTrue(entry.substring(resolve, normalize).contains("if (cudaStr == nullptr)"));
        assertTrue(entry.substring(resolve, normalize).contains(
                "LaunchContext::defaultContext()->getCudaStream()"));
        assertEquals(resolve, entry.lastIndexOf("cudaStream_t cudaStr ="),
                "Context and forwarded pointer must use one canonical handle");
        for (String dispatch : new String[] {"segDispatchCompile(", "segDispatchCaptureOrDirect("}) {
            assertTrue(entry.indexOf(dispatch) > normalize, dispatch);
        }
        assertTrue(entry.indexOf("ctx.cudaStr = cudaStr;") > normalize);
    }

    @Test
    void stagingFailureUnwindsCaptureBeforeMarkingFailure() throws Exception {
        String source = source();
        int start = source.indexOf("const auto abortStagingCapture = [&]()");
        int end = source.indexOf("NDArray** staged = stagingResult.effectiveExternals;", start);
        assertTrue(start >= 0 && end > start);
        String staging = source.substring(start, end);
        assertTrue(staging.contains("abortCapture(seg, true, didPushCtx, tritonCaptureDevice,"));
        assertTrue(staging.contains("prevCaptureStream, savedSlotPhasesTriton, stream);"));
        assertTrue(staging.contains("TritonGraphBackend::clearOrderedRangeExecutor();"));
        int failure = staging.indexOf("if (!stagingResult.ok() || stagingResult.effectiveExternals == nullptr)");
        String failed = staging.substring(failure);
        assertTrue(failed.indexOf("abortStagingCapture();") < failed.indexOf("SegmentLifecycle::markFailed("));
        assertTrue(failed.contains("return setGpuBackendFailureDetail("));
        assertFalse(failed.contains("return Status::OK"));
        String throwing = staging.substring(staging.indexOf("} catch (...) {"), failure);
        assertTrue(throwing.contains("abortStagingCapture();"));
        assertTrue(throwing.contains("throw;"), "Never swallow the original staging exception");
        assertTrue(throwing.contains("capture cleanup also failed"));
    }
}
