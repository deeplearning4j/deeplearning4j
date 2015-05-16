/*
 *
 *  * Copyright 2015 Skymind,Inc.
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
 *
 */

package org.nd4j.linalg.api.ops.exception;

/**
 * Thrown with illegal operations
 *
 * @author Adam Gibson
 */
public class IllegalOpException extends Exception {
    public IllegalOpException() {
    }

    public IllegalOpException(String message) {
        super(message);
    }

    public IllegalOpException(String message, Throwable cause) {
        super(message, cause);
    }

    public IllegalOpException(Throwable cause) {
        super(cause);
    }

    public IllegalOpException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
