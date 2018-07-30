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

@javax.annotation.Generated(value = {"org.deeplearning4j.ui.stats.sbe.StatsType"})
public enum StatsType {
    Parameters((short) 0), Gradients((short) 1), Updates((short) 2), Activations((short) 3), NULL_VAL((short) 255);

    private final short value;

    StatsType(final short value) {
        this.value = value;
    }

    public short value() {
        return value;
    }

    public static StatsType get(final short value) {
        switch (value) {
            case 0:
                return Parameters;
            case 1:
                return Gradients;
            case 2:
                return Updates;
            case 3:
                return Activations;
        }

        if ((short) 255 == value) {
            return NULL_VAL;
        }

        throw new IllegalArgumentException("Unknown value: " + value);
    }
}
