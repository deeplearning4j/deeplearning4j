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

/** Source boundary guards; CUDA numerical coverage is run separately with VERIFY/full. */
@Tag("source-lint")
class DspPhaseCompileProbeTest {
    private String source(String name) throws Exception {
        return Files.readString(Path.of("..").toAbsolutePath().normalize().resolve(
                "libnd4j/include/graph/impl/" + name));
    }

    @Test
    void observationsBracketCompileBeforeCaptureCleanup() throws Exception {
        String source = source("NativeDynamicShapePlan.cpp");
        int before = source.indexOf("\"before-phaseCompile\"");
        int compile = source.indexOf("phaseCompile(externalInputs, numExternalInputs);", before);
        int after = source.indexOf("\"after-phaseCompile\"", compile);
        int cleanup = source.indexOf("dspEndStaleCapture", after);
        assertTrue(before >= 0 && compile > before && after > compile && cleanup > after);
        String boundary = source.substring(source.lastIndexOf("#ifdef SD_CUDA", before), after);
        assertEquals(2, boundary.split("probePhaseCompileOutputs\\(", -1).length - 1);
        assertEquals(2, boundary.split("requestedOutputs, requestedOutputSlotIndices_", -1).length - 1);
    }

    @Test
    void probeUsesOwnedDeviceSnapshotAndCaptureSafeExplicitOptIn() throws Exception {
        String source = source("NativeDynamicShapePlan_cuda.cu");
        int start = source.indexOf("void summarizePhaseCompileOutput(");
        int end = source.indexOf("void NativeDynamicShapePlan::platformDumpLogitsArgmax", start);
        assertTrue(start >= 0 && end > start);
        String probe = source.substring(start, end);
        assertTrue(probe.contains("std::vector<T> snapshot"));
        assertTrue(probe.contains("static_cast<const T*>(db->special()) + lo"));
        assertTrue(probe.contains("snapshot[arr->getOffset(i) - lo]"));
        assertTrue(probe.contains("cudaMemcpyDeviceToHost, copyStream"));
        assertTrue(probe.contains("cudaStreamSynchronize(copyStream)"));
        assertFalse(probe.contains("->syncToHost("));
        assertFalse(probe.contains("->primary("));
        assertFalse(probe.contains("->e<"));
        int entry = probe.indexOf("void probePhaseCompileOutputs(");
        String driver = probe.substring(entry);
        assertTrue(driver.indexOf("getEnabledMask() & DSP_DIAG_VERIFY") < driver.indexOf("std::vector<cudaStream_t>"));
        assertTrue(driver.contains("getLevel() != DSP_LEVEL_FULL) return;"));
        assertTrue(driver.indexOf("cudaStreamIsCapturing(candidate, &status)") < driver.indexOf("cudaStreamSynchronize(candidate)"));
        assertTrue(driver.contains("DebugHelper::inGraphCapture(nullptr)"));
        assertFalse(driver.contains("cudaGetLastError("));
        assertFalse(driver.contains("dspEndStaleCapture("));
        assertTrue(driver.contains("SD_FLOAT_TYPES"));
    }
}
