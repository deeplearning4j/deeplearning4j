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
public enum OrtErrorCode {
    ORT_OK(0),
    ORT_FAIL(1),
    ORT_INVALID_ARGUMENT(2),
    ORT_NO_SUCHFILE(3),
    ORT_NO_MODEL(4),
    ORT_ENGINE_ERROR(5),
    ORT_RUNTIME_EXCEPTION(6),
    ORT_INVALID_PROTOBUF(7),
    ORT_MODEL_LOADED(8),
    ORT_NOT_IMPLEMENTED(9),
    ORT_INVALID_GRAPH(10),
    ORT_EP_FAIL(11);


    private final int methodIndex;

    OrtErrorCode(int index) {
        this.methodIndex = index;
    }
}