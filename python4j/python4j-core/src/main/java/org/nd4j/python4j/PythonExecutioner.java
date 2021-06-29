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

import org.bytedeco.cpython.PyObject;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.commons.io.IOUtils;
import org.bytedeco.cpython.PyThreadState;
import org.bytedeco.cpython.global.python;
import org.nd4j.common.io.ClassPathResource;

import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.cpython.helper.python.Py_SetPath;

/**
 * PythonExecutioner handles executing python code either from passed in python code
 * or via a python script.
 *
 * PythonExecutioner has a few java system properties to be aware of when executing python:
 * @link {{@link #DEFAULT_PYTHON_PATH_PROPERTY}} : The default python path to be used by the executioner.
 * This can be passed with -Dorg.eclipse.python4j.path=your/python/path
 *
 * Python4j has a default python path that imports the javacpp python path depending on what is present.
 * The javacpp python presets such as {@link org.bytedeco.cpython.global.python} have a cachePackages() method
 * that leverages loading python artifacts from the python path.
 *
 * This python path can be merged with a custom one or just used as is.
 * A user specifies this behavior with the system property {@link #JAVACPP_PYTHON_APPEND_TYPE}
 * This property can have 3 possible values:
 * 1. before
 * 2. after
 * 3. none
 *
 * Order can matter when resolving versions of libraries very similar to the system path. Ensure when adding a custom python path
 * that these properties are well tested and well understood before use.
 *
 * @author Adam Gibson, Fariz Rahman
 */
public class PythonExecutioner {
    private final static String PYTHON_EXCEPTION_KEY = "__python_exception__";
    private static AtomicBoolean init = new AtomicBoolean(false);
    public final static String DEFAULT_PYTHON_PATH_PROPERTY = "org.eclipse.python4j.path";
    public final static String JAVACPP_PYTHON_APPEND_TYPE = "org.eclipse.python4j.path.append";
    public final static String DEFAULT_APPEND_TYPE = "before";
    public final static String INITIALIZE_PYTHON = "org.eclipse.python4j.python.initialize";
    public final static String DEFAULT_INITIALIZE_PYTHON = "true";
    static {
        init();
    }

    private static synchronized void init() {
        if (init.get()) {
            return;
        }

        init.set(true);
        initPythonPath();
        if(Boolean.parseBoolean(System.getProperty(INITIALIZE_PYTHON,DEFAULT_INITIALIZE_PYTHON)))
            Py_InitializeEx(0);
        //initialize separately to ensure that numpy import array is not imported twice
        for (PythonType type: PythonTypes.get()) {
            type.init();
        }

        //set the main thread state for the gil
        PythonGIL.setMainThreadState();
        PyEval_SaveThread();

    }

    /**
     * Sets a variable.
     *
     * @param name
     * @param value
     */
    public static void setVariable(String name, PythonObject value) {
        PythonGIL.assertThreadSafe();
        PyObject main = PyImport_ImportModule("__main__");
        PyObject globals = PyModule_GetDict(main);
        PyDict_SetItemString(globals, name, value.getNativePythonObject());
        Py_DecRef(main);

    }

    /**
     * Sets given list of PythonVariables in the interpreter.
     *
     * @param pyVars
     */
    public static void setVariables(List<PythonVariable> pyVars) {
        for (PythonVariable pyVar : pyVars)
            setVariable(pyVar.getName(), pyVar.getPythonObject());
    }

    /**
     * Sets given list of PythonVariables in the interpreter.
     *
     * @param pyVars
     */
    public static void setVariables(PythonVariable... pyVars) {
        setVariables(Arrays.asList(pyVars));
    }

    /**
     * Gets the given list of PythonVariables from the interpreter.
     *
     * @param pyVars
     */
    public static void getVariables(List<PythonVariable> pyVars) {
        for (PythonVariable pyVar : pyVars)
            pyVar.setValue(getVariable(pyVar.getName(), pyVar.getType()).getValue());
    }

    /**
     * Gets the given list of PythonVariables from the interpreter.
     *
     * @param pyVars
     */
    public static void getVariables(PythonVariable... pyVars) {
        getVariables(Arrays.asList(pyVars));
    }



    /**
     * Gets the variable with the given name from the interpreter.
     *
     * @param name
     * @return
     */
    public static PythonObject getVariable(String name) {
        PythonGIL.assertThreadSafe();
        PyObject main = PyImport_ImportModule("__main__");
        PyObject globals = PyModule_GetDict(main);
        PyObject pyName = PyUnicode_FromString(name);
        try {
            if (PyDict_Contains(globals, pyName) == 1) {
                return new PythonObject(PyObject_GetItem(globals, pyName), false);
            }
        } finally {
            Py_DecRef(main);
            //Py_DecRef(globals);
            Py_DecRef(pyName);
        }
        return new PythonObject(null);
    }

    /**
     * Gets the variable with the given name from the interpreter.
     *
     * @param name
     * @return
     */
    public static <T> PythonVariable<T> getVariable(String name, PythonType<T> type) {
        PythonObject val = getVariable(name);
        return new PythonVariable<>(name, type, type.toJava(val));
    }

    /**
     * Executes a string of code
     *
     * @param code
     */
    public static synchronized void simpleExec(String code) {
        PythonGIL.assertThreadSafe();

        int result = PyRun_SimpleStringFlags(code, null);
        if (result != 0) {
            throw new PythonException("Execution failed, unable to retrieve python exception.");
        }
    }

    private static void throwIfExecutionFailed() {
        PythonObject ex = getVariable(PYTHON_EXCEPTION_KEY);
        if (ex != null && !ex.isNone() && !ex.toString().isEmpty()) {
            setVariable(PYTHON_EXCEPTION_KEY, PythonTypes.STR.toPython(""));
            throw new PythonException(ex);
        }
    }


    private static String getWrappedCode(String code) {
        ClassPathResource resource = new ClassPathResource("org/nd4j/python4j/pythonexec/pythonexec.py");
        try (InputStream is = resource.getInputStream()) {
            String base = IOUtils.toString(is, StandardCharsets.UTF_8);
            String indentedCode = "    " + code.replace("\n", "\n    ");
            String out = base.replace("    pass", indentedCode);
            return out;
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read python code!", e);
        }

    }

    /**
     * Executes a string of code. Throws PythonException if execution fails.
     *
     * @param code
     */
    public static void exec(String code) {
        simpleExec(getWrappedCode(code));
        throwIfExecutionFailed();
    }

    public static void exec(String code, List<PythonVariable> inputs, List<PythonVariable> outputs) {
        if (inputs != null) {
            setVariables(inputs.toArray(new PythonVariable[0]));
        }
        exec(code);
        if (outputs != null) {
            getVariables(outputs.toArray(new PythonVariable[0]));
        }
    }

    /**
     * Return list of all supported variables in the interpreter.
     *
     * @return
     */
    public static PythonVariables getAllVariables() {
        PythonGIL.assertThreadSafe();
        PythonVariables ret = new PythonVariables();
        PyObject main = PyImport_ImportModule("__main__");
        PyObject globals = PyModule_GetDict(main);
        PyObject keys = PyDict_Keys(globals);
        PyObject keysIter = PyObject_GetIter(keys);
        try {

            long n = PyObject_Size(globals);
            for (int i = 0; i < n; i++) {
                PyObject pyKey = PyIter_Next(keysIter);
                try {
                    if (!new PythonObject(pyKey, false).toString().startsWith("_")) {

                        PyObject pyVal = PyObject_GetItem(globals, pyKey); // TODO check ref count
                        PythonType pt;
                        try {
                            pt = PythonTypes.getPythonTypeForPythonObject(new PythonObject(pyVal, false));

                        } catch (PythonException pe) {
                            pt = null;
                        }
                        if (pt != null) {
                            ret.add(
                                    new PythonVariable<>(
                                            new PythonObject(pyKey, false).toString(),
                                            pt,
                                            pt.toJava(new PythonObject(pyVal, false))
                                    )
                            );
                        }
                    }
                } finally {
                    Py_DecRef(pyKey);
                }
            }
        } finally {
            Py_DecRef(keysIter);
            Py_DecRef(keys);
            Py_DecRef(main);
            return ret;
        }

    }


    /**
     * Executes a string of code and returns a list of all supported variables.
     *
     * @param code
     * @param inputs
     * @return
     */
    public static PythonVariables execAndReturnAllVariables(String code, List<PythonVariable> inputs) {
        setVariables(inputs);
        simpleExec(getWrappedCode(code));
        return getAllVariables();
    }

    /**
     * Executes a string of code and returns a list of all supported variables.
     *
     * @param code
     * @return
     */
    public static PythonVariables execAndReturnAllVariables(String code) {
        simpleExec(getWrappedCode(code));
        return getAllVariables();
    }

    private static synchronized void initPythonPath() {
        try {
            String path = System.getProperty(DEFAULT_PYTHON_PATH_PROPERTY);

            List<File> packagesList = new ArrayList<>();
            packagesList.addAll(Arrays.asList(cachePackages()));
            for (PythonType type: PythonTypes.get()){
                packagesList.addAll(Arrays.asList(type.packages()));
            }
            //// TODO: fix in javacpp
            packagesList.add(new File(python.cachePackage(), "site-packages"));

            File[] packages = packagesList.toArray(new File[0]);

            if (path == null) {
                Py_AddPath(packages);
            } else {
                StringBuffer sb = new StringBuffer();

                JavaCppPathType pathAppendValue = JavaCppPathType.valueOf(System.getProperty(JAVACPP_PYTHON_APPEND_TYPE, DEFAULT_APPEND_TYPE).toUpperCase());
                switch (pathAppendValue) {
                    case BEFORE:
                        for (File cacheDir : packages) {
                            sb.append(cacheDir);
                            sb.append(java.io.File.pathSeparator);
                        }

                        sb.append(path);
                        break;
                    case AFTER:
                        sb.append(path);

                        for (File cacheDir : packages) {
                            sb.append(cacheDir);
                            sb.append(java.io.File.pathSeparator);
                        }
                        break;
                    case NONE:
                        sb.append(path);
                        break;
                }

                Py_AddPath(sb.toString());
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private enum JavaCppPathType {
        BEFORE, AFTER, NONE
    }

    private static File[] cachePackages() throws IOException {
        File[] path = org.bytedeco.cpython.global.python.cachePackages();
        path = Arrays.copyOf(path, path.length + 1);
        path[path.length - 1] = cachePackage();
        return path;
    }

}