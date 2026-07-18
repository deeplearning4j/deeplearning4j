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
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

/**
 * A model-level audio synthesis request. Transport, authentication and artifact
 * retention deliberately remain the responsibility of the serving application.
 */
public final class AudioSynthesisRequest {

    private final UUID runId;
    private final String text;
    private final String voice;
    private final String language;
    private final Map<String, Object> configuration;

    public AudioSynthesisRequest(UUID runId, String text, String voice, String language,
                                 Map<String, Object> configuration) {
        this.runId = Objects.requireNonNull(runId, "runId");
        if (text == null || text.trim().isEmpty()) {
            throw new IllegalArgumentException("text must not be blank");
        }
        this.text = text;
        this.voice = voice == null ? "" : voice;
        this.language = language == null ? "" : language;
        this.configuration = immutableMap(configuration);
    }

    public UUID getRunId() {
        return runId;
    }

    public String getText() {
        return text;
    }

    public String getVoice() {
        return voice;
    }

    public String getLanguage() {
        return language;
    }

    public Map<String, Object> getConfiguration() {
        return configuration;
    }

    private static Map<String, Object> immutableMap(Map<String, Object> source) {
        if (source == null || source.isEmpty()) {
            return Collections.emptyMap();
        }
        Map<String, Object> copy = new LinkedHashMap<>();
        source.forEach((key, value) -> copy.put(
                Objects.requireNonNull(key, "configuration key"), immutableValue(value)));
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
}
