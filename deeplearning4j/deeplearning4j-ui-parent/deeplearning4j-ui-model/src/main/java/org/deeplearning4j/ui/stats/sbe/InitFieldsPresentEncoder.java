/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/*- Generated SBE (Simple Binary Encoding) message codec */
package org.deeplearning4j.ui.stats.sbe;

import org.agrona.MutableDirectBuffer;

@javax.annotation.Generated(value = {"org.deeplearning4j.ui.stats.sbe.InitFieldsPresentEncoder"})
@SuppressWarnings("all")
public class InitFieldsPresentEncoder {
    public static final int ENCODED_LENGTH = 1;
    private MutableDirectBuffer buffer;
    private int offset;

    public InitFieldsPresentEncoder wrap(final MutableDirectBuffer buffer, final int offset) {
        this.buffer = buffer;
        this.offset = offset;

        return this;
    }

    public int encodedLength() {
        return ENCODED_LENGTH;
    }

    public InitFieldsPresentEncoder clear() {
        buffer.putByte(offset, (byte) (short) 0);
        return this;
    }

    public InitFieldsPresentEncoder softwareInfo(final boolean value) {
        byte bits = buffer.getByte(offset);
        bits = (byte) (value ? bits | (1 << 0) : bits & ~(1 << 0));
        buffer.putByte(offset, bits);
        return this;
    }

    public InitFieldsPresentEncoder hardwareInfo(final boolean value) {
        byte bits = buffer.getByte(offset);
        bits = (byte) (value ? bits | (1 << 1) : bits & ~(1 << 1));
        buffer.putByte(offset, bits);
        return this;
    }

    public InitFieldsPresentEncoder modelInfo(final boolean value) {
        byte bits = buffer.getByte(offset);
        bits = (byte) (value ? bits | (1 << 2) : bits & ~(1 << 2));
        buffer.putByte(offset, bits);
        return this;
    }
}
