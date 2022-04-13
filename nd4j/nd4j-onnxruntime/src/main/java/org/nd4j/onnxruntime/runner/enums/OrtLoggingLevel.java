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
package org.nd4j.onnxruntime.runner.enums;

/**
 * Onnx Type referenced here:
 * https://onnxruntime.ai/docs/api/c/group___global.html
 * This contains  the known types for inputs and outputs
 * for various onnx data structures.
 *
 * @author Adam Gibson
 */
public enum OrtLoggingLevel {
    ONNX_LOGGING_LEVEL_VERBOSE(0),
    ONNX_LOGGING_LEVEL_INFO(1),
    ONNX_LOGGING_LEVEL_WARNING(2),
    ONNX_LOGGING_LEVEL_ERROR(3),
    ONNX_LOGGING_LEVEL_FATAL(4);


    private final int methodIndex;

    OrtLoggingLevel(int index) {
        this.methodIndex = index;
    }
}