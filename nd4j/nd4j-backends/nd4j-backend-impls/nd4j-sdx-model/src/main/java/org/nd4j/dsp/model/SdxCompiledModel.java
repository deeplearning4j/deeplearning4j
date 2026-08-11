/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

/**
 * Opaque resolved result of one canonical SDZ for one target profile.
 *
 * <p>{@link #runtimeModelPath()} is intentionally the only provider-facing path
 * consumers normally need. Its physical suffix is an internal compiler-cache
 * detail, not a model format selected by the application.</p>
 */
public final class SdxCompiledModel {
    private final Path sourceModel;
    private final Path cacheEntry;
    private final Path runtimeModelPath;
    private final Path tokenizerPath;
    private final Path tokenizerConfigPath;
    private final Path textGenerationConfigPath;
    private final Path quantizationConfigPath;
    private final SdxSourceIdentity sourceIdentity;
    private final SdxTargetProfile target;
    private final String compileKey;
    private final String compilerId;
    private final String compilerVersion;

    SdxCompiledModel(
            Path sourceModel,
            Path cacheEntry,
            Path runtimeModelPath,
            Path tokenizerPath,
            Path tokenizerConfigPath,
            Path textGenerationConfigPath,
            Path quantizationConfigPath,
            SdxSourceIdentity sourceIdentity,
            SdxTargetProfile target,
            String compileKey,
            String compilerId,
            String compilerVersion) {
        this.sourceModel = Objects.requireNonNull(sourceModel, "sourceModel");
        this.cacheEntry = Objects.requireNonNull(cacheEntry, "cacheEntry");
        this.runtimeModelPath = Objects.requireNonNull(runtimeModelPath, "runtimeModelPath");
        this.tokenizerPath = tokenizerPath;
        this.tokenizerConfigPath = tokenizerConfigPath;
        this.textGenerationConfigPath = textGenerationConfigPath;
        this.quantizationConfigPath = quantizationConfigPath;
        this.sourceIdentity = Objects.requireNonNull(sourceIdentity, "sourceIdentity");
        this.target = Objects.requireNonNull(target, "target");
        this.compileKey = Objects.requireNonNull(compileKey, "compileKey");
        this.compilerId = Objects.requireNonNull(compilerId, "compilerId");
        this.compilerVersion = Objects.requireNonNull(compilerVersion, "compilerVersion");
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

    public Optional<Path> tokenizerPath() {
        return Optional.ofNullable(tokenizerPath);
    }

    /**
     * Returns the Hugging Face-compatible tokenizer configuration snapshot used
     * when this cache entry was compiled. It carries special-token and chat-template
     * semantics that are not represented by {@code tokenizer.json} itself.
     */
    public Optional<Path> tokenizerConfigPath() {
        return Optional.ofNullable(tokenizerConfigPath);
    }

    /**
     * Validates and returns the complete immutable asset set required by a text
     * generation consumer. Generic graph consumers may continue to use the
     * optional accessors above.
     */
    public SdxTextModelAssets requireTextModelAssets() throws IOException {
        return SdxTextModelAssets.require(this);
    }

    public Optional<Path> textGenerationConfigPath() {
        return Optional.ofNullable(textGenerationConfigPath);
    }

    public Optional<Path> quantizationConfigPath() {
        return Optional.ofNullable(quantizationConfigPath);
    }

    public SdxSourceIdentity sourceIdentity() {
        return sourceIdentity;
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
}
