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

package org.datavec.api.io.converters;

/**
 * Writable converter exception represents an error
 * for being unable to convert a writable
 * @author Adam Gibson
 */
public class WritableConverterException extends Exception {
    public WritableConverterException() {}

    public WritableConverterException(String message) {
        super(message);
    }

    public WritableConverterException(String message, Throwable cause) {
        super(message, cause);
    }

    public WritableConverterException(Throwable cause) {
        super(cause);
    }

    public WritableConverterException(String message, Throwable cause, boolean enableSuppression,
                    boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
