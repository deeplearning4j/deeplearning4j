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

package org.nd4j.linalg.jtpu;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.cpu.nativecpu.NDArray;

/**
 * @deprecated TPU uses the standard host-native {@link NDArray}. This subtype
 * remains as a source-compatibility bridge; PJRT device buffers are owned by
 * native replay handles rather than individual Java arrays.
 */
@Deprecated
public class JTpuNDArray extends NDArray {
    public JTpuNDArray() { super(); }
    public JTpuNDArray(DataBuffer buffer) { super(buffer); }
    public JTpuNDArray(DataBuffer buffer, long[] shape) { super(buffer, shape); }
    public JTpuNDArray(DataBuffer buffer, long[] shape, long[] stride,
                       long offset, char ordering) {
        super(buffer, shape, stride, offset, ordering);
    }
    public JTpuNDArray(DataBuffer buffer, long[] shape, long[] stride,
                       long offset, long elementWiseStride, char ordering,
                       boolean view) {
        super(buffer, shape, stride, offset, elementWiseStride, ordering, view);
    }
    public JTpuNDArray(float[] data, long[] shape, char ordering) {
        super(data, shape, 0, ordering);
    }
    public JTpuNDArray(double[] data, long[] shape, char ordering) {
        super(data, shape, ordering);
    }

    /** PJRT uploads are scoped to compiled segment replay, not array ownership. */
    public boolean isOnDevice() { return false; }

    /** Retained for compatibility; the next PJRT replay uploads this host value. */
    public JTpuNDArray toDevice() { return this; }

    /** Values exposed by this class are already host-native. */
    public JTpuNDArray toHost() { return this; }
}
