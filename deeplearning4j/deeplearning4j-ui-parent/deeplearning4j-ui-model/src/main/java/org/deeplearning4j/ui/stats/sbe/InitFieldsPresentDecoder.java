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

import org.agrona.DirectBuffer;

@javax.annotation.Generated(value = {"org.deeplearning4j.ui.stats.sbe.InitFieldsPresentDecoder"})
@SuppressWarnings("all")
public class InitFieldsPresentDecoder {
    public static final int ENCODED_LENGTH = 1;
    private DirectBuffer buffer;
    private int offset;

    public InitFieldsPresentDecoder wrap(final DirectBuffer buffer, final int offset) {
        this.buffer = buffer;
        this.offset = offset;

        return this;
    }

    public int encodedLength() {
        return ENCODED_LENGTH;
    }

    public boolean softwareInfo() {
        return 0 != (buffer.getByte(offset) & (1 << 0));
    }

    public boolean hardwareInfo() {
        return 0 != (buffer.getByte(offset) & (1 << 1));
    }

    public boolean modelInfo() {
        return 0 != (buffer.getByte(offset) & (1 << 2));
    }

    public String toString() {
        return appendTo(new StringBuilder(100)).toString();
    }

    public StringBuilder appendTo(final StringBuilder builder) {
        builder.append('{');
        boolean atLeastOne = false;
        if (softwareInfo()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("softwareInfo");
            atLeastOne = true;
        }
        if (hardwareInfo()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("hardwareInfo");
            atLeastOne = true;
        }
        if (modelInfo()) {
            if (atLeastOne) {
                builder.append(',');
            }
            builder.append("modelInfo");
            atLeastOne = true;
        }
        builder.append('}');

        return builder;
    }
}
