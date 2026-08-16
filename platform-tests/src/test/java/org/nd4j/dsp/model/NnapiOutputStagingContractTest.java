/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class NnapiOutputStagingContractTest {

    @Test
    void everyOutputUsesTheCompiledDescriptorAndOwnedStorageUntilNnapiCompletes()
            throws Exception {
        Path backend = Path.of("")
                .toAbsolutePath()
                .normalize()
                .resolve("../libnd4j/include/graph/cpu/NnapiGraphBackend.cpp")
                .normalize();
        Path backendHeader = backend.resolveSibling("NnapiGraphBackend.h");
        assertTrue(Files.isRegularFile(backend), "NNAPI backend source was not found at " + backend);
        assertTrue(Files.isRegularFile(backendHeader),
                "NNAPI backend header was not found at " + backendHeader);

        String source = Files.readString(backend);
        String header = Files.readString(backendHeader);
        int descriptorValidation = source.indexOf(
                "matchesCompiledDescriptor(arr, mapping.sourceDataType,");
        int staging = source.indexOf(
                "auto staging = std::make_unique<NDArray>(",
                descriptorValidation);
        int binding = source.indexOf(
                "void* buffer = boundOutput->buffer();",
                staging);
        int wait = source.indexOf(
                "result = ANeuralNetworksEvent_wait(event);",
                binding);
        int copyBack = source.indexOf(
                "arr->assign(boundOutput);",
                wait);

        assertTrue(header.contains("DataType sourceDataType;")
                        && header.contains("DataType bindingDataType;")
                        && header.contains("std::vector<LongType> dimensions;"),
                "Compiled NNAPI mappings must retain their source and binding descriptors");
        assertTrue(descriptorValidation >= 0,
                "Live DSP output metadata must match the compiled operand descriptor");
        assertTrue(staging > descriptorValidation,
                "Every output must receive an independent compiled-descriptor staging array");
        assertTrue(binding > staging, "NNAPI must bind the owned staging buffer");
        assertTrue(wait > binding, "The staging buffer must survive until the NNAPI event completes");
        assertTrue(copyBack > wait, "Staging must copy back only after successful completion");
        assertTrue(!source.contains("NDArray* boundOutput = arr;"),
                "NNAPI must not write directly into mutable DynamicShapePlan arrays");
        assertTrue(
                source.contains("NNAPI_OUTPUT_STAGING seg[%d-%d] output=%u source_slot=%d"),
                "Detailed DSP diagnostics must identify the output and source slot");
        assertTrue(
                source.contains("boundOutput->lengthOf() * boundOutput->sizeOfT()"),
                "NNAPI output capacity must be derived from the bound staging array");
    }

    @Test
    void recurrentStateCrossesThePrefillDecodeBoundaryThroughHostOwnedStorage()
            throws Exception {
        Path session = Path.of("")
                .toAbsolutePath()
                .normalize()
                .resolve("../libnd4j/include/legacy/impl/SdxGenerationSession.cpp")
                .normalize();
        assertTrue(Files.isRegularFile(session), "SDX generation source was not found at " + session);

        String source = Files.readString(session);
        int copyStart = source.indexOf("bool copyRecurrentArray(");
        int copyEnd = source.indexOf("\n}\n\n}  // namespace", copyStart);
        assertTrue(copyStart >= 0 && copyEnd > copyStart,
                "The recurrent-state transfer helper must remain explicit");
        String transfer = source.substring(copyStart, copyEnd);

        assertTrue(
                transfer.contains("source->syncToHost();"),
                "Borrowed accelerator output must be synchronized before leaving the prefill context");
        assertTrue(
                transfer.contains("void* sourceBuffer = source->buffer();"),
                "The transfer must validate that the prefill output exposes host storage");
        assertTrue(
                transfer.contains("void* destinationBuffer = destination->buffer();"),
                "The decode state must expose independently owned host storage");
        assertTrue(
                transfer.contains("std::memcpy(destinationBuffer, sourceBuffer, bytes);"),
                "Recurrent state must cross the context boundary through an explicit host copy");
        assertTrue(
                transfer.contains("destination->tickWriteHost();"),
                "The copied decode state must be marked host-authoritative");
        assertTrue(
                !transfer.contains("destination->assign(source);"),
                "Generic assignment must not retain device-only ownership across the public ABI boundary");
    }
}
