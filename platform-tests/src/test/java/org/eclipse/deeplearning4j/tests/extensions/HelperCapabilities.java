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

package org.eclipse.deeplearning4j.tests.extensions;

/**
 * Helper capabilities for feature detection.
 */
public class HelperCapabilities {
    private final Helper helper;
    private final boolean available;
    private final boolean supportsFloat16;
    private final boolean supportsBFloat16;
    private final boolean supportsInt8;
    private final boolean supportsConvolution;
    private final boolean supportsRNN;
    private final boolean supportsAttention;
    private final boolean supportsBatchNorm;
    private final String version;

    public HelperCapabilities(Helper helper, boolean available) {
        this(helper, available, false, false, false, true, true, true, true, "unknown");
    }

    public HelperCapabilities(Helper helper, boolean available, boolean supportsFloat16,
                              boolean supportsBFloat16, boolean supportsInt8,
                              boolean supportsConvolution, boolean supportsRNN,
                              boolean supportsAttention, boolean supportsBatchNorm,
                              String version) {
        this.helper = helper;
        this.available = available;
        this.supportsFloat16 = supportsFloat16;
        this.supportsBFloat16 = supportsBFloat16;
        this.supportsInt8 = supportsInt8;
        this.supportsConvolution = supportsConvolution;
        this.supportsRNN = supportsRNN;
        this.supportsAttention = supportsAttention;
        this.supportsBatchNorm = supportsBatchNorm;
        this.version = version;
    }

    public Helper getHelper() { return helper; }
    public boolean isAvailable() { return available; }
    public boolean supportsFloat16() { return supportsFloat16; }
    public boolean supportsBFloat16() { return supportsBFloat16; }
    public boolean supportsInt8() { return supportsInt8; }
    public boolean supportsConvolution() { return supportsConvolution; }
    public boolean supportsRNN() { return supportsRNN; }
    public boolean supportsAttention() { return supportsAttention; }
    public boolean supportsBatchNorm() { return supportsBatchNorm; }
    public String getVersion() { return version; }

    @Override
    public String toString() {
        return String.format("HelperCapabilities{helper=%s, available=%s, version=%s}",
                helper, available, version);
    }
}
