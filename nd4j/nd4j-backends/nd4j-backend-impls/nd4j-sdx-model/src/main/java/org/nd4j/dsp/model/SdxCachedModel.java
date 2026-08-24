/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Metadata-only view of one validated source/target reference in an {@link SdxModelCache}.
 *
 * <p>Inventory does not hash multi-gigabyte payloads. It verifies the immutable manifest,
 * validation marker, contained paths, and declared file sizes. {@link SdxModelCache#resolve}
 * remains the authoritative full validation boundary before execution.</p>
 */
public final class SdxCachedModel {
    private final Path sourceModel;
    private final Path cacheEntry;
    private final Path runtimeModelPath;
    private final String sourceSha256;
    private final long sourceLogicalBytes;
    private final long sourcePhysicalBytes;
    private final String sourceFileName;
    private final SdxTargetProfile target;
    private final String compileKey;
    private final String compilerId;
    private final String compilerVersion;
    private final long objectPhysicalBytes;
    private final long lastModifiedMillis;

    SdxCachedModel(
            Path sourceModel,
            Path cacheEntry,
            Path runtimeModelPath,
            String sourceSha256,
            long sourceLogicalBytes,
            long sourcePhysicalBytes,
            String sourceFileName,
            SdxTargetProfile target,
            String compileKey,
            String compilerId,
            String compilerVersion,
            long objectPhysicalBytes,
            long lastModifiedMillis) {
        this.sourceModel = Objects.requireNonNull(sourceModel, "sourceModel");
        this.cacheEntry = Objects.requireNonNull(cacheEntry, "cacheEntry");
        this.runtimeModelPath = Objects.requireNonNull(runtimeModelPath, "runtimeModelPath");
        this.sourceSha256 = Objects.requireNonNull(sourceSha256, "sourceSha256");
        this.sourceLogicalBytes = sourceLogicalBytes;
        this.sourcePhysicalBytes = sourcePhysicalBytes;
        this.sourceFileName = Objects.requireNonNull(sourceFileName, "sourceFileName");
        this.target = Objects.requireNonNull(target, "target");
        this.compileKey = Objects.requireNonNull(compileKey, "compileKey");
        this.compilerId = Objects.requireNonNull(compilerId, "compilerId");
        this.compilerVersion = Objects.requireNonNull(compilerVersion, "compilerVersion");
        this.objectPhysicalBytes = objectPhysicalBytes;
        this.lastModifiedMillis = lastModifiedMillis;
    }

    public Path sourceModel() {
        return sourceModel;
    }

    public Path cacheEntry() {
        return cacheEntry;
    }

    public Path runtimeModelPath() {
        return runtimeModelPath;
    }

    public String sourceSha256() {
        return sourceSha256;
    }

    public long sourceLogicalBytes() {
        return sourceLogicalBytes;
    }

    public long sourcePhysicalBytes() {
        return sourcePhysicalBytes;
    }

    public String sourceFileName() {
        return sourceFileName;
    }

    public SdxTargetProfile target() {
        return target;
    }

    public String compileKey() {
        return compileKey;
    }

    public String compilerId() {
        return compilerId;
    }

    public String compilerVersion() {
        return compilerVersion;
    }

    public long objectPhysicalBytes() {
        return objectPhysicalBytes;
    }

    public long referencedPhysicalBytes() {
        return Math.addExact(sourcePhysicalBytes, objectPhysicalBytes);
    }

    public long lastModifiedMillis() {
        return lastModifiedMillis;
    }
}
