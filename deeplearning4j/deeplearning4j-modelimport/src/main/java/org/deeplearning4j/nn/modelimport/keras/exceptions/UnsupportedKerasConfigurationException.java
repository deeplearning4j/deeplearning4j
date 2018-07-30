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

package org.deeplearning4j.nn.modelimport.keras.exceptions;


/**
 * Indicates that user is attempting to import a Keras model configuration that
 * is not currently supported.
 *
 * See http://deeplearning4j.org/model-import-keras for more information and
 * file an issue at http://github.com/deeplearning4j/deeplearning4j/issues.
 *
 * @author dave@skymind.io
 */
public class UnsupportedKerasConfigurationException extends Exception {

    public UnsupportedKerasConfigurationException(String message) {
        super(appendDocumentationURL(message));
    }

    public UnsupportedKerasConfigurationException(String message, Throwable cause) {
        super(appendDocumentationURL(message), cause);
    }

    public UnsupportedKerasConfigurationException(Throwable cause) {
        super(cause);
    }

    private static String appendDocumentationURL(String message) {
        return message + ". Please file an issue at http://github.com/deeplearning4j/deeplearning4j/issues.";
    }
}
