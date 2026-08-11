/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Complete immutable inputs required to open an offline SDX causal-language-model session.
 *
 * <p>This is deliberately stricter than {@link SdxCompiledModel}: image, audio, and generic
 * graph consumers do not need text assets, while chat applications must never discover a
 * missing tokenizer or prompt contract after the provider session has already opened.</p>
 */
public final class SdxTextModelAssets {
    private final Path tokenizer;
    private final Path tokenizerConfig;
    private final Path textGenerationConfig;

    private SdxTextModelAssets(
            Path tokenizer,
            Path tokenizerConfig,
            Path textGenerationConfig) {
        this.tokenizer = tokenizer;
        this.tokenizerConfig = tokenizerConfig;
        this.textGenerationConfig = textGenerationConfig;
    }

    public static SdxTextModelAssets require(SdxCompiledModel model) throws IOException {
        Objects.requireNonNull(model, "model");
        Path cacheEntry = model.cacheEntry().toAbsolutePath().normalize();
        Path tokenizer = requireAsset(
                cacheEntry,
                model.tokenizerPath().orElse(null),
                "tokenizer.json");
        Path tokenizerConfig = requireAsset(
                cacheEntry,
                model.tokenizerConfigPath().orElse(null),
                "tokenizer configuration (tokenizer_config.json)");
        Path textGenerationConfig = requireAsset(
                cacheEntry,
                model.textGenerationConfigPath().orElse(null),
                "SDX text-generation graph contract");
        return new SdxTextModelAssets(tokenizer, tokenizerConfig, textGenerationConfig);
    }

    public Path tokenizer() {
        return tokenizer;
    }

    public Path tokenizerConfig() {
        return tokenizerConfig;
    }

    public Path textGenerationConfig() {
        return textGenerationConfig;
    }

    private static Path requireAsset(Path cacheEntry, Path candidate, String label)
            throws IOException {
        if (candidate == null) {
            throw new IOException(
                    "Compiled SDX is not a runnable text model: missing " + label
                            + ". Rebuild the SDZ with the required text-model assets.");
        }
        Path asset = candidate.toAbsolutePath().normalize();
        if (!asset.startsWith(cacheEntry)
                || Files.isSymbolicLink(asset)
                || !Files.isRegularFile(asset, LinkOption.NOFOLLOW_LINKS)
                || Files.size(asset) <= 0L) {
            throw new IOException(
                    "Compiled SDX is not a runnable text model: " + label
                            + " is missing, empty, unsafe, or outside its immutable cache object: "
                            + asset);
        }
        return asset;
    }
}
