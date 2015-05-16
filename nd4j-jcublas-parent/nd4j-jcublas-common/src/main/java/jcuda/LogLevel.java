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

package jcuda;

/**
 * The log levels which may be used to control the internal
 * logging of the JCuda libraries
 */
public enum LogLevel
{
    /**
     * Never print anything
     */
    LOG_QUIET,

    /**
     * Only print error messages
     */
    LOG_ERROR,

    /**
     * Print warnings
     */
    LOG_WARNING,

    /**
     * Print info messages
     */
    LOG_INFO,

    /**
     * Print debug information
     */
    LOG_DEBUG,

    /**
     * Trace function calls
     */
    LOG_TRACE,

    /**
     * Print fine-grained debug information
     */
    LOG_DEBUGTRACE
}
