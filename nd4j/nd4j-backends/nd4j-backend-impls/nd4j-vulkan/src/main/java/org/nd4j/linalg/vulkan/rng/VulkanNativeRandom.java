/*
 * ******************************************************************************
 * *
 * * This program and the accompanying materials are made available under the
 * * terms of the Apache License, Version 2.0 which is available at
 * * https://www.apache.org/licenses/LICENSE-2.0.
 * *
 * * SPDX-License-Identifier: Apache-2.0
 * *****************************************************************************
 */
package org.nd4j.linalg.vulkan.rng;

import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.linalg.vulkan.VulkanRuntime;
import org.nd4j.nativeblas.OpaqueRandomGenerator;
import org.nd4j.rng.NativeRandom;

/**
 * Vulkan native random-generator state.
 *
 * <p>Scalar state transitions use libnd4j's backend-neutral native generator.
 * Array generation still goes through the Vulkan execution path and is rejected
 * unless a Vulkan device operation is available.</p>
 */
public final class VulkanNativeRandom extends NativeRandom {
    public VulkanNativeRandom() {
        this(System.currentTimeMillis());
    }

    public VulkanNativeRandom(long seed) {
        super(seed);
    }

    public VulkanNativeRandom(long seed, long numberOfElements) {
        super(seed, numberOfElements);
    }

    @Override
    public void init() {
        nativeOps = VulkanRuntime.getInstance().nativeOps();
        statePointer = nativeOps.createRandomGenerator(seed, seed ^ 0xdeadbeefL);
        if (nativeOps.lastErrorCode() != 0) {
            throw new IllegalStateException(nativeOps.lastErrorMessage());
        }
        setSeed(seed);
    }

    @Override
    public void close() {
        if (statePointer != null && !statePointer.isNull()) {
            nativeOps.deleteRandomGenerator((OpaqueRandomGenerator) statePointer);
            statePointer.setNull();
        }
    }

    @Override
    public PointerPointer getExtraPointers() {
        return null;
    }

    @Override
    public void setSeed(long seed) {
        this.seed = seed;
        currentPosition.set(0);
        nativeOps.setRandomGeneratorStates(
                (OpaqueRandomGenerator) statePointer, seed, seed ^ 0xdeadbeefL);
    }

    @Override
    public long getSeed() {
        return seed;
    }

    @Override
    public float nextFloat() {
        return nativeOps.getRandomGeneratorNextFloat((OpaqueRandomGenerator) statePointer);
    }

    @Override
    public double nextDouble() {
        return nativeOps.getRandomGeneratorNextDouble((OpaqueRandomGenerator) statePointer);
    }

    @Override
    public int nextInt() {
        return nativeOps.getRandomGeneratorNextInt((OpaqueRandomGenerator) statePointer);
    }

    @Override
    public long nextLong() {
        return nativeOps.getRandomGeneratorNextLong((OpaqueRandomGenerator) statePointer);
    }

    public long rootState() {
        return nativeOps.getRandomGeneratorRootState((OpaqueRandomGenerator) statePointer);
    }

    public long nodeState() {
        return nativeOps.getRandomGeneratorNodeState((OpaqueRandomGenerator) statePointer);
    }

    @Override
    public void setStates(long rootState, long nodeState) {
        nativeOps.setRandomGeneratorStates(
                (OpaqueRandomGenerator) statePointer, rootState, nodeState);
    }
}
