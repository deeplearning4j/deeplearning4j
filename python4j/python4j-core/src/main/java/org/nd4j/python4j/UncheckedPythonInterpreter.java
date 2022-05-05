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

import org.apache.commons.io.IOUtils;
import org.bytedeco.cpython.PyObject;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.io.ClassPathResource;

import java.io.InputStream;
import java.nio.charset.Charset;

/**
 * A port of UncheckedPythonWrapper from:
 * https://github.com/invesdwin/invesdwin-context-python/blob/master/invesdwin-context-python-parent/invesdwin-context-python-runtime-python4j/src/main/java/de/invesdwin/context/python/runtime/python4j/internal/UncheckedPythonEngineWrapper.java#L125
 *
 *
 */
public class UncheckedPythonInterpreter implements PythonInterpreter {


    private static final String ANS = "__ans__";
    private static final String ANS_EQUALS = ANS + " = ";

    private static UncheckedPythonInterpreter INSTANCE;

    private static PyObject globals;
    private static PyObject globalsAns;

    private final GILLock gilLock = new GILLock();



    static {
        INSTANCE = new UncheckedPythonInterpreter();
    }


    private UncheckedPythonInterpreter() {
    }

    public PythonObject newNone() {
        evalUnchecked(ANS_EQUALS + "None");
        return getAns();
    }

    public void init() {
        synchronized (UncheckedPythonInterpreter.class) {
            PythonExecutioner.init();
            if (UncheckedPythonInterpreter.globals != null) {
                return;
            }

            gilLock.lock();

            try (InputStream is = new ClassPathResource(UncheckedPythonInterpreter.class.getSimpleName() + ".py").getInputStream()) {
                String code = IOUtils.toString(is, Charset.defaultCharset());
                final int result = org.bytedeco.cpython.global.python.PyRun_SimpleString(code);
                if (result != 0) {
                    throw new PythonException("Execution failed, unable to retrieve python exception.");
                }


                final PyObject main = org.bytedeco.cpython.global.python.PyImport_ImportModule("__main__");
                UncheckedPythonInterpreter.globals = org.bytedeco.cpython.global.python.PyModule_GetDict(main);
                UncheckedPythonInterpreter.globalsAns = org.bytedeco.cpython.global.python.PyUnicode_FromString(ANS);
                //we keep the refs eternally
                //org.bytedeco.cpython.global.python.Py_DecRef(main);
                //org.bytedeco.cpython.global.python.Py_DecRef(globals);
                //org.bytedeco.cpython.global.python.Py_DecRef(globalsAns);

            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                gilLock.unlock();

            }
        }
    }

    private void eval(final String expression) {
        gilLock.lock();
        try {
            evalUnchecked(expression);
        } finally {
            gilLock.unlock();
        }
    }

    private void evalUnchecked(final String expression) {
        final int result = org.bytedeco.cpython.global.python.PyRun_SimpleString(expression);
        if (result != 0) {
            throw new PythonException("Execution failed, unable to retrieve python exception.");
        }
    }

    private PythonObject getAns() {
        return new PythonObject(org.bytedeco.cpython.global.python.PyObject_GetItem(globals, globalsAns), false);
    }


    public static UncheckedPythonInterpreter getInstance() {
        return UncheckedPythonInterpreter.INSTANCE;
    }

    @Override
    public void exec(String expression) {
        eval(expression);
    }

    @Override
    public Object get(String variable) {
        gilLock.lock();
        try {
            evalUnchecked(ANS_EQUALS + variable);
            final PythonObject ans = getAns();
            final PythonType<Object> type = PythonTypes.getPythonTypeForPythonObject(ans);
            return type.toJava(ans);
        } finally {
            gilLock.unlock();
        }
    }

    @Override
    public void set(String variable, Object value) {
        gilLock.lock();
        try {
            if (value == null) {
                evalUnchecked(variable + " = None");
            } else {
                final PythonObject converted = PythonTypes.convert(value);
                org.bytedeco.cpython.global.python.PyDict_SetItemString(globals, variable,
                        converted.getNativePythonObject());
            }
        } catch (final Throwable t) {
            throw new RuntimeException("Variable=" + variable + " Value=" + value, t);
        } finally {
            gilLock.unlock();
        }
    }
}
