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

import org.nd4j.linalg.cpu.nativecpu.ops.CpuOpContext;

/**
 * TPU backend op context.
 *
 * <p>The standard opaque native context is backend-neutral and is also the
 * context used to compile and execute DSP plans. PJRT objects remain owned by
 * native replay handles, so no raw PJRT lifetime crosses into Java.</p>
 */
public final class TpuOpContext extends CpuOpContext {
    public TpuOpContext() {
        super();
    }
}
