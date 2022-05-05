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

import org.nd4j.common.primitives.AtomicBoolean;

/**
 * A port of:
 * https://github.com/invesdwin/invesdwin-context-python/blob/master/invesdwin-context-python-parent/invesdwin-context-python-runtime-python4j/src/main/java/de/invesdwin/context/python/runtime/python4j/internal/InitializingPythonEngineWrapper.java
 * Permission given here:
 * https://github.com/eclipse/deeplearning4j/issues/9595
 *
 * @author Adam Gibson
 */
public class InitializingPythonInterpreter implements PythonInterpreter {

    private static AtomicBoolean initialized = new AtomicBoolean(false);
    private final PythonInterpreter delegate;

    public InitializingPythonInterpreter() {
        delegate = UncheckedPythonInterpreter.getInstance();
    }

    public static void maybeInit() {
        if (initialized.get()) {
            return;
        }
        synchronized (InitializingPythonInterpreter.class) {
            if (initialized.get()) {
                return;
            }

            UncheckedPythonInterpreter.getInstance().init();
            initialized.set(true);
        }
    }

    @Override
    public void exec(String expression) {
        delegate.exec(expression);
    }

    @Override
    public Object get(String variable) {
        InitializingPythonInterpreter.maybeInit();
        return delegate.get(variable);
    }

    @Override
    public void set(String variable, Object value) {
        InitializingPythonInterpreter.maybeInit();

        delegate.set(variable,value);
    }

    public static PythonInterpreter getInstance() {
        InitializingPythonInterpreter.maybeInit();
        return UncheckedPythonInterpreter.getInstance();
    }
}
