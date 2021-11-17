/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.api.ops.impl.layers.convolution.config;


public enum PaddingMode {
    VALID(0),
    SAME(1),
    CAUSAL(2);

    public final int index;
    PaddingMode(int index) { this.index = index; }

    public static PaddingMode fromNumber(int index) {
        switch(index) {
            case 0: return VALID;
            case 1: return SAME;
            case 2: return CAUSAL;
            default:throw new IllegalArgumentException("Illegal index passed in: " + index);
        }

    }
}
