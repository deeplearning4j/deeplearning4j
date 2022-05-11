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
package org.nd4j.python4j;


import org.nd4j.common.primitives.Pair;

/**
 * Base interface for a python interpreter for running python code.
 * Based on:
 * https://github.com/invesdwin/invesdwin-context-python/blob/master/invesdwin-context-python-parent/invesdwin-context-python-runtime-python4j/src/main/java/de/invesdwin/context/python/runtime/python4j/internal/IPythonEngineWrapper.java
 *
 * Permission given here: https://github.com/eclipse/deeplearning4j/issues/9595
 *
 * @author Adam Gibson
 */
public interface PythonInterpreter {


    Object getCachedPython(String varName);

    Object getCachedJava(String varName);

    Pair<PythonObject,Object> getCachedPythonJava(String varName);

    /**
     * GIL Lock object
     * @return
     */
    GILLock gilLock();

    /**
     * Execute the given python code
     * @param expression a python expression
     */
    void exec(String expression);

    /**
     * Retrieve a variable from the interpreter.
     * Returns none if none is found or variable
     * does not exist
     *
     * @param variable the variable to retrieve
     * @param getNew
     * @return null if the object is none or the variable does not exist
     */
    Object get(String variable, boolean getNew);

    /**
     * Set a variable in the interpreter
     * @param variable the variable to set
     * @param value the value to set
     */
    void set(String variable, Object value);


}
