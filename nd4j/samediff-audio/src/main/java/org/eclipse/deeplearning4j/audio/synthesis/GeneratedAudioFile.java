/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */

package org.eclipse.deeplearning4j.audio.synthesis;

import java.lang.reflect.Array;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Describes a completed model-generated audio file.
 *
 * <p>The generator returns a file instead of an in-memory byte array. The
 * serving process is expected to validate containment, compute the digest and
 * byte length, and publish its own opaque transfer reference.</p>
 */
public final class GeneratedAudioFile {

    private final Path completedFile;
    private final String mediaType;
    private final String modelId;
    private final String modelVersion;
    private final String configurationVersion;
    private final double confidence;
    private final Map<String, Object> configurationEvidence;

    public GeneratedAudioFile(Path completedFile, String mediaType, String modelId,
                              String modelVersion, String configurationVersion,
                              double confidence, Map<String, Object> configurationEvidence) {
        this.completedFile = Objects.requireNonNull(completedFile, "completedFile");
        this.mediaType = requireNonBlank(mediaType, "mediaType");
        if (!this.mediaType.toLowerCase(Locale.ROOT).startsWith("audio/")) {
            throw new IllegalArgumentException("mediaType must be audio/*");
        }
        this.modelId = requireNonBlank(modelId, "modelId");
        this.modelVersion = requireNonBlank(modelVersion, "modelVersion");
        this.configurationVersion = requireNonBlank(configurationVersion, "configurationVersion");
        if (!Double.isFinite(confidence) || confidence < 0.0 || confidence > 1.0) {
            throw new IllegalArgumentException("confidence must be between 0 and 1");
        }
        this.confidence = confidence;
        this.configurationEvidence = immutableMap(configurationEvidence);
    }

    public Path getCompletedFile() {
        return completedFile;
    }

    public String getMediaType() {
        return mediaType;
    }

    public String getModelId() {
        return modelId;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public String getConfigurationVersion() {
        return configurationVersion;
    }

    public double getConfidence() {
        return confidence;
    }

    public Map<String, Object> getConfigurationEvidence() {
        return configurationEvidence;
    }

    private static Map<String, Object> immutableMap(Map<String, Object> source) {
        if (source == null || source.isEmpty()) {
            return Collections.emptyMap();
        }
        Map<String, Object> copy = new LinkedHashMap<>();
        source.forEach((key, value) -> copy.put(
                Objects.requireNonNull(key, "configuration evidence key"), immutableValue(value)));
        return Collections.unmodifiableMap(copy);
    }

    private static Object immutableValue(Object value) {
        if (value instanceof Map<?, ?>) {
            Map<?, ?> source = (Map<?, ?>) value;
            Map<Object, Object> copy = new LinkedHashMap<>();
            source.forEach((key, nested) -> copy.put(key, immutableValue(nested)));
            return Collections.unmodifiableMap(copy);
        }
        if (value instanceof Collection<?>) {
            Collection<?> source = (Collection<?>) value;
            List<Object> copy = new ArrayList<>(source.size());
            source.forEach(nested -> copy.add(immutableValue(nested)));
            return Collections.unmodifiableList(copy);
        }
        if (value != null && value.getClass().isArray()) {
            int length = Array.getLength(value);
            List<Object> copy = new ArrayList<>(length);
            for (int index = 0; index < length; index++) {
                copy.add(immutableValue(Array.get(value, index)));
            }
            return Collections.unmodifiableList(copy);
        }
        return value;
    }

    private static String requireNonBlank(String value, String name) {
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException(name + " must not be blank");
        }
        return value;
    }
}
