/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.jtpu.ops;

import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BroadcastOp;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.RandomOp;
import org.nd4j.linalg.api.ops.ReduceOp;
import org.nd4j.linalg.api.ops.ScalarOp;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.cpu.nativecpu.ops.NativeOpExecutioner;
import org.nd4j.nativeblas.NativeOps;

import java.util.Properties;

/**
 * TPU executioner backed by StableHLO/PJRT.
 *
 * <p>Custom/eager descriptors and SameDiff DSP segments share one native
 * trait/KernelSpec/StableHLO lowering path. Legacy numbered op families have no
 * canonical KernelSpec identity yet and fail explicitly instead of executing
 * host CPU numerics.</p>
 */
public final class TpuExecutioner extends NativeOpExecutioner {

    public TpuExecutioner() {
        super();
    }

    public TpuExecutioner(NativeOps nativeOps, boolean secondary) {
        super(nativeOps, secondary);
    }

    @Override
    public OpContext buildContext() {
        return new TpuOpContext();
    }

    private UnsupportedOperationException unsupportedLegacy(Op op) {
        return new UnsupportedOperationException(
                "TPU eager execution requires a canonical CustomOp/KernelSpec; legacy op "
                        + (op == null ? "<null>" : op.opName()) + " is not lowerable");
    }

    @Override public INDArray exec(Op op) { throw unsupportedLegacy(op); }
    @Override public INDArray exec(Op op, OpContext context) { throw unsupportedLegacy(op); }
    @Override public INDArray exec(ScalarOp op) { throw unsupportedLegacy(op); }
    @Override public INDArray exec(ReduceOp op) { throw unsupportedLegacy(op); }
    @Override public INDArray exec(Variance op) { throw unsupportedLegacy(op); }
    @Override public INDArray exec(IndexAccumulation op) { throw unsupportedLegacy(op); }
    @Override public INDArray exec(BroadcastOp op) { throw unsupportedLegacy(op); }
    @Override public INDArray exec(RandomOp op, Random random) { throw unsupportedLegacy(op); }

    @Override
    public ExecutionerType type() {
        return ExecutionerType.TPU;
    }

    @Override
    public Properties getEnvironmentInformation() {
        Properties properties = super.getEnvironmentInformation();
        properties.setProperty("backend", "TPU");
        properties.setProperty("runtime", "PJRT");
        properties.setProperty("compiled.graph.format", "StableHLO");
        properties.setProperty("bfloat16.support", "true");
        return properties;
    }
}
