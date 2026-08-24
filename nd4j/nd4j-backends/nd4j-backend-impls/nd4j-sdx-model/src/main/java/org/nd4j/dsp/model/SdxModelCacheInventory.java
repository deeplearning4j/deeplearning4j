/*
 * Copyright (c) Eclipse Deeplearning4j
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.model;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Immutable storage snapshot for one SDX model cache. */
public final class SdxModelCacheInventory {
    private final List<SdxCachedModel> entries;
    private final long totalPhysicalBytes;
    private final long referencedSourceBytes;
    private final long referencedObjectBytes;
    private final int invalidReferenceCount;

    SdxModelCacheInventory(
            List<SdxCachedModel> entries,
            long totalPhysicalBytes,
            long referencedSourceBytes,
            long referencedObjectBytes,
            int invalidReferenceCount) {
        this.entries = Collections.unmodifiableList(new ArrayList<>(entries));
        this.totalPhysicalBytes = totalPhysicalBytes;
        this.referencedSourceBytes = referencedSourceBytes;
        this.referencedObjectBytes = referencedObjectBytes;
        this.invalidReferenceCount = invalidReferenceCount;
    }

    public List<SdxCachedModel> entries() {
        return entries;
    }

    /** Every regular byte under the cache root, including indexes and unreferenced data. */
    public long totalPhysicalBytes() {
        return totalPhysicalBytes;
    }

    /** Unique canonical source bytes referenced by the returned entries. */
    public long referencedSourceBytes() {
        return referencedSourceBytes;
    }

    /** Unique immutable object bytes referenced by the returned entries. */
    public long referencedObjectBytes() {
        return referencedObjectBytes;
    }

    public int invalidReferenceCount() {
        return invalidReferenceCount;
    }
}
