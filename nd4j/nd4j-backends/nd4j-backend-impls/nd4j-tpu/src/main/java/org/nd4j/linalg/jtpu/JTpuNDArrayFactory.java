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

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.cpu.nativecpu.CpuNDArrayFactory;

/**
 * @deprecated TPU uses {@link CpuNDArrayFactory} as its host-native control
 * plane. This subclass preserves the former public construction surface.
 */
@Deprecated
public class JTpuNDArrayFactory extends CpuNDArrayFactory {
    public JTpuNDArrayFactory() { super(); }
    public JTpuNDArrayFactory(DataType dataType, Character order) {
        super(dataType, order);
    }
    public JTpuNDArrayFactory(DataType dataType, char order) {
        super(dataType, order);
    }
}
