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

public class DeepLearningException extends Exception {

    /**
     * 
     */
    private static final long serialVersionUID = -7973589163269627293L;

    public DeepLearningException() {
        super();
    }

    public DeepLearningException(String message, Throwable cause, boolean enableSuppression,
                    boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }

    public DeepLearningException(String message, Throwable cause) {
        super(message, cause);
    }

    public DeepLearningException(String message) {
        super(message);
    }

    public DeepLearningException(Throwable cause) {
        super(cause);
    }



}
