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
 * Available helper implementations.
 */
public enum Helper {
    CPU("cpu", "Native CPU implementation"),
    ONEDNN("onednn", "Intel OneDNN (MKL-DNN)"),
    CUDNN("cudnn", "NVIDIA cuDNN"),
    ARMCOMPUTE("armcompute", "ARM Compute Library"),
    MPS("mps", "Apple Metal Performance Shaders"),
    ACCELERATE("accelerate", "Apple Accelerate Framework"),
    MLIR("mlir", "MLIR/LLVM JIT Compilation"),
    MIOPEN("miopen", "AMD MIOpen"),
    PJRT("pjrt", "PJRT/TPU");

    private final String id;
    private final String description;

    Helper(String id, String description) {
        this.id = id;
        this.description = description;
    }

    public String getId() { return id; }
    public String getDescription() { return description; }

    public static Helper fromString(String str) {
        if (str == null) return null;
        String normalized = str.toLowerCase().trim();
        // Handle aliases
        if (normalized.equals("mkldnn") || normalized.equals("dnnl")) {
            normalized = "onednn";
        }
        for (Helper h : values()) {
            if (h.id.equals(normalized)) {
                return h;
            }
        }
        return null;
    }
}
