/*-
 *
 *  * Copyright 2016 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

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
