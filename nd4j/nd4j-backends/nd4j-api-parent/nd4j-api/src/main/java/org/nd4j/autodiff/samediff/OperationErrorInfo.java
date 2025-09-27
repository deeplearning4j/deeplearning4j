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

package org.nd4j.autodiff.samediff;

import lombok.Data;

/**
 * Information about operation execution errors
 */
@Data
public class OperationErrorInfo {
    /**
     * Error message describing what went wrong
     */
    private String errorMessage;
    
    /**
     * Type/class of the error that occurred
     */
    private String errorType;
    
    /**
     * Timestamp when the error occurred
     */
    private long timestamp;
    
    /**
     * Full stack trace of the error
     */
    private String stackTrace;
    
    /**
     * Context information when the error occurred
     */
    private String errorContext;
    
    /**
     * Whether this error is recoverable
     */
    private boolean recoverable = false;
    
    /**
     * Suggested recovery actions
     */
    private String recoveryActions;
    
    /**
     * Error severity level
     */
    private ErrorSeverity severity = ErrorSeverity.ERROR;
    
    /**
     * Additional error metadata
     */
    private java.util.Map<String, Object> errorMetadata = new java.util.HashMap<>();
    
    /**
     * Constructor for basic error info
     */
    public OperationErrorInfo(String errorMessage, String errorType) {
        this.errorMessage = errorMessage;
        this.errorType = errorType;
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * Default constructor
     */
    public OperationErrorInfo() {
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * Enumeration of error severity levels
     */
    public enum ErrorSeverity {
        WARNING,
        ERROR,
        CRITICAL,
        FATAL
    }
}
