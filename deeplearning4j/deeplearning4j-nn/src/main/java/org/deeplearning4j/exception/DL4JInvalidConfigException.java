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

package org.deeplearning4j.exception;

/**
 * Exception signifying that the specified configuration is invalid
 *
 * @author Alex Black
 */
public class DL4JInvalidConfigException extends DL4JException {
    public DL4JInvalidConfigException() {}

    public DL4JInvalidConfigException(String message) {
        super(message);
    }

    public DL4JInvalidConfigException(String message, Throwable cause) {
        super(message, cause);
    }

    public DL4JInvalidConfigException(Throwable cause) {
        super(cause);
    }
}
