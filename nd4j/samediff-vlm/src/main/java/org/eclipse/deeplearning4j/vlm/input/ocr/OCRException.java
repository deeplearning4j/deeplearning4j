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

package org.eclipse.deeplearning4j.vlm.input.ocr;

/**
 * Exception thrown when OCR processing fails.
 *
 * Adam Gibson
 */
public class OCRException extends Exception {

    private final ErrorCode errorCode;

    public OCRException(String message) {
        super(message);
        this.errorCode = ErrorCode.UNKNOWN;
    }

    public OCRException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = ErrorCode.UNKNOWN;
    }

    public OCRException(ErrorCode errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    public OCRException(ErrorCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }

    public ErrorCode getErrorCode() {
        return errorCode;
    }

    /**
     * OCR error codes.
     */
    public enum ErrorCode {
        /**
         * Unknown error.
         */
        UNKNOWN,

        /**
         * Engine not initialized.
         */
        NOT_INITIALIZED,

        /**
         * Invalid image format.
         */
        INVALID_IMAGE,

        /**
         * Image too large.
         */
        IMAGE_TOO_LARGE,

        /**
         * Model not found.
         */
        MODEL_NOT_FOUND,

        /**
         * Language not supported.
         */
        LANGUAGE_NOT_SUPPORTED,

        /**
         * GPU not available.
         */
        GPU_NOT_AVAILABLE,

        /**
         * Processing timeout.
         */
        TIMEOUT,

        /**
         * Out of memory.
         */
        OUT_OF_MEMORY,

        /**
         * Engine-specific error.
         */
        ENGINE_ERROR
    }
}
