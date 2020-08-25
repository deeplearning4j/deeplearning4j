
/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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


package org.datavec.python;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.IOUtils;
import org.bytedeco.cpython.global.python;
import org.bytedeco.numpy.global.numpy;
import org.nd4j.common.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.bytedeco.cpython.global.python.*;
import static org.datavec.python.Python.*;

/**
 *  Allows execution of python scripts managed by
 *  an internal interpreter.
 *  An end user may specify a python script to run
 *  via any of the execution methods available in this class.
 *
 *  At static initialization time (when the class is first initialized)
 *  a number of components are setup:
 *  1. The python path. A user may over ride this with the system property {@link #DEFAULT_PYTHON_PATH_PROPERTY}
 *
 *  2. Since this executioner uses javacpp to manage and run python interpreters underneath the covers,
 *  a user may also over ride the system property {@link #JAVACPP_PYTHON_APPEND_TYPE} with one of the {@link JavaCppPathType}
 *  values. This will allow the user to determine whether the javacpp default python path is used at all, and if so
 *  whether it is appended, prepended, or not used. This behavior is useful when you need to use an external
 *  python distribution such as anaconda.
 *
 *  3. A proper numpy import for use with javacpp: We call numpy import ourselves to ensure proper loading of
 *  native libraries needed by numpy are allowed to load in the proper order. If we don't do this,
 *  it causes a variety of issues with running numpy. (User must still include "import numpy as np" in their scripts).
 *
 *  4. Various python scripts pre defined on the classpath included right with the java code.
 *  These are auxillary python scripts used for loading classes, pre defining certain kinds of behavior
 *  in order for us to manipulate values within the python memory, as well as pulling them out of memory
 *  for integration within the internal python executioner.
 *
 *  For more information on how this works, please take a look at the {@link #init()}
 *  method.
 *
 *  Generally, a user defining a python script for use by the python executioner
 *  will have a set of defined target input values and output values.
 *  These values should not be present when actually running the script, but just referenced.
 *  In order to test your python script for execution outside the engine,
 *  we recommend commenting out a few default values as dummy input values.
 *  This will allow an end user to test their script before trying to use the server.
 *
 *  In order to get output values out of a python script, all a user has to do
 *  is define the output variables they want being used in the final output in the actual pipeline.
 *  For example, if a user wants to return a dictionary, they just have to create a dictionary with that name
 *  and based on the configured {@link PythonVariables} passed as outputs
 *  to one of the execution methods, we can pull the values out automatically.
 *
 *  For input definitions, it is similar. You just define the values you want used in
 *  {@link PythonVariables} and we will automatically generate code for defining those values
 *  as desired for running. This allows the user to customize values dynamically
 *  at runtime but reference them by name in a python script.
 *
 *
 *  @author Fariz Rahman
 *  @author Adam Gibson
 */


@Slf4j
public class PythonExecutioner {


    private static AtomicBoolean init = new AtomicBoolean(false);
    public final static String DEFAULT_PYTHON_PATH_PROPERTY = "org.datavec.python.path";
    public final static String JAVACPP_PYTHON_APPEND_TYPE = "org.datavec.python.javacpp.path.append";
    public final static String DEFAULT_APPEND_TYPE = "before";
    private final static String PYTHON_EXCEPTION_KEY = "__python_exception__";

    static {
        init();
    }


    private static synchronized void init() {
        if (init.get()) {
            return;
        }
        initPythonPath();
        init.set(true);
        log.info("CPython: PyEval_InitThreads()");
        PyEval_InitThreads();
        log.info("CPython: Py_InitializeEx()");
        Py_InitializeEx(0);
        numpy._import_array();
    }

    private static synchronized void simpleExec(String code) throws PythonException{
        log.debug(code);
        log.info("CPython: PyRun_SimpleStringFlag()");

        int result = PyRun_SimpleStringFlags(code, null);
        if (result != 0) {
            throw new PythonException("Execution failed, unable to retrieve python exception.");
        }
    }

    public static boolean validateVariableName(String s) {
        if (s.isEmpty()) return false;
        if (!Character.isJavaIdentifierStart(s.charAt(0))) return false;
        for (int i = 1; i < s.length(); i++)
            if (!Character.isJavaIdentifierPart(s.charAt(i)))
                return false;
        return true;
    }


    /**
     * Sets a variable in the global scope of the current context (See @PythonContextManager).
     * This is equivalent to `exec("a = b");` where a is the variable name
     * and b is the variable value.
     * @param varName Name of the python variable being set. Should be a valid python identifier string
     * @param pythonObject Value for the python variable
     * @throws Exception
     */
    public static void setVariable(String varName, PythonObject pythonObject) throws PythonException{
        if (!validateVariableName(varName)){
            throw new PythonException("Invalid variable name: " + varName);
        }
        Python.globals().set(new PythonObject(varName), pythonObject);
    }

    public  static <T> void setVariable(String varName, PythonType varType, Object value) throws PythonException {
        PythonObject pythonObject;
        switch (varType.getName()) {
            case STR:
                pythonObject = new PythonObject(PythonType.STR.convert(value));
                break;
            case INT:
                pythonObject = new PythonObject(PythonType.INT.convert(value));
                break;
            case FLOAT:
                pythonObject = new PythonObject(PythonType.FLOAT.convert(value));
                break;
            case BOOL:
                pythonObject = new PythonObject(PythonType.BOOL.convert(value));
                break;
            case NDARRAY:
                pythonObject = new PythonObject(PythonType.NDARRAY.convert(value));
                break;
            case LIST:
                pythonObject = new PythonObject(PythonType.LIST.convert(value));
                break;
            case DICT:
                pythonObject = new PythonObject(PythonType.DICT.convert(value));
                break;
            case BYTES:
                pythonObject = new PythonObject(PythonType.BYTES.convert(value));
                break;
            default:
                throw new PythonException("Unsupported type: " + varType);

        }
        setVariable(varName, pythonObject);
    }

    public static void setVariables(PythonVariables pyVars) throws PythonException{
        if (pyVars == null) return;
        for (String varName : pyVars.getVariables()) {
            setVariable(varName, pyVars.getType(varName), pyVars.getValue(varName));
        }
    }

    public static PythonObject getVariable(String varName) {
        return Python.globals().attr("get").call(varName);
    }

    public static <T> T getVariable(String varName, PythonType<T> varType) throws PythonException{
        PythonObject pythonObject = getVariable(varName);
        return varType.toJava(pythonObject);
    }

    public static void getVariables(PythonVariables pyVars) throws PythonException {
        if (pyVars == null){
            return;
        }
        for (String varName : pyVars.getVariables()) {
            pyVars.setValue(varName, getVariable(varName, pyVars.getType(varName)));
        }
    }


    private static String getWrappedCode(String code) {
        try (InputStream is = new ClassPathResource("pythonexec/pythonexec.py").getInputStream()) {
            String base = IOUtils.toString(is, Charset.defaultCharset());
            StringBuffer indentedCode = new StringBuffer();
            for (String split : code.split("\n")) {
                indentedCode.append("    " + split + "\n");

            }

            String out = base.replace("    pass", indentedCode);
            return out;
        } catch (IOException e) {
            throw new IllegalStateException("Unable to read python code!", e);
        }

    }

    private static void throwIfExecutionFailed() throws PythonException{
        PythonObject ex = getVariable(PYTHON_EXCEPTION_KEY);
        if (ex != null && !ex.isNone() && !ex.toString().isEmpty()) {
            setVariable(PYTHON_EXCEPTION_KEY, new PythonObject(""));
            throw new PythonException(ex);
        }
    }

    public static void exec(String code) throws PythonException {
        simpleExec(getWrappedCode(code));
        throwIfExecutionFailed();
    }

    public static void exec(String code, PythonVariables inputVariables, PythonVariables outputVariables) throws PythonException {
        setVariables(inputVariables);
        simpleExec(getWrappedCode(code));
        throwIfExecutionFailed();
        getVariables(outputVariables);
    }

    public static PythonVariables execAndReturnAllVariables(String code) throws PythonException {
        simpleExec(getWrappedCode(code));
        throwIfExecutionFailed();
        PythonVariables out = new PythonVariables();
        PythonObject globals = Python.globals();
        PythonObject keysList = Python.list(globals.attr("keys"));
        int numKeys = Python.len(keysList).toInt();
        for (int i = 0; i < numKeys; i++) {
            PythonObject key = keysList.get(i);
            String keyStr = key.toString();
            if (!keyStr.startsWith("_")) {
                PythonObject val = globals.get(key);
                if (Python.isinstance(val, intType())) {
                    out.addInt(keyStr, val.toInt());
                } else if (Python.isinstance(val, floatType())) {
                    out.addFloat(keyStr, val.toDouble());
                } else if (Python.isinstance(val, strType())) {
                    out.addStr(keyStr, val.toString());
                } else if (Python.isinstance(val, boolType())) {
                    out.addBool(keyStr, val.toBoolean());
                } else if (Python.isinstance(val, listType())) {
                    out.addList(keyStr, val.toList().toArray(new Object[0]));
                } else if (Python.isinstance(val, dictType())) {
                    out.addDict(keyStr, val.toMap());
                }
            }
        }
        return out;

    }

    public static PythonVariables getAllVariables() throws PythonException{
        PythonVariables out = new PythonVariables();
        PythonObject globals = Python.globals();
        PythonObject keysList = Python.list(globals.attr("keys").call());
        int numKeys = Python.len(keysList).toInt();
        for (int i = 0; i < numKeys; i++) {
            PythonObject key = keysList.get(i);
            String keyStr = key.toString();
            if (!keyStr.startsWith("_")) {
                PythonObject val = globals.get(key);
                if (Python.isinstance(val, intType())) {
                    out.addInt(keyStr, val.toInt());
                } else if (Python.isinstance(val, floatType())) {
                    out.addFloat(keyStr, val.toDouble());
                } else if (Python.isinstance(val, strType())) {
                    out.addStr(keyStr, val.toString());
                } else if (Python.isinstance(val, boolType())) {
                    out.addBool(keyStr, val.toBoolean());
                } else if (Python.isinstance(val, listType())) {
                    out.addList(keyStr, val.toList().toArray(new Object[0]));
                } else if (Python.isinstance(val, dictType())) {
                    out.addDict(keyStr, val.toMap());
                } else {
                    PythonObject np = importModule("numpy");
                    if (Python.isinstance(val, np.attr("ndarray"), np.attr("generic"))) {
                        out.addNDArray(keyStr, val.toNumpy());
                    }
                }

            }
        }
        return out;
    }

    public static PythonVariables execAndReturnAllVariables(String code, PythonVariables inputs) throws Exception{
        setVariables(inputs);
        simpleExec(getWrappedCode(code));
        return getAllVariables();
    }

    /**
     * One of a few desired values
     * for how we should handle
     * using javacpp's python path.
     * BEFORE: Prepend the python path alongside a defined one
     * AFTER: Append the javacpp python path alongside the defined one
     * NONE: Don't use javacpp's python path at all
     */
    public enum JavaCppPathType {
        BEFORE, AFTER, NONE
    }

    /**
     * Set the python path.
     * Generally you can just use the PYTHONPATH environment variable,
     * but if you need to set it from code, this can work as well.
     */

    public static synchronized void initPythonPath() {
        if (!init.get()) {
            try {
                String path = System.getProperty(DEFAULT_PYTHON_PATH_PROPERTY);
                if (path == null) {
                    log.info("Setting python default path");
                    File[] packages = numpy.cachePackages();

                    //// TODO: fix in javacpp
                    File sitePackagesWindows = new File(python.cachePackage(), "site-packages");
                    File[] packages2 = new File[packages.length + 1];
                    System.arraycopy(packages, 0, packages2, 0, packages.length);
                    packages2[packages.length] = sitePackagesWindows;
                    //System.out.println(sitePackagesWindows.getAbsolutePath());
                    packages = packages2;
                    //////////

                    Py_SetPath(packages);
                } else {
                    log.info("Setting python path " + path);
                    StringBuffer sb = new StringBuffer();
                    File[] packages = numpy.cachePackages();
                    JavaCppPathType pathAppendValue = JavaCppPathType.valueOf(System.getProperty(JAVACPP_PYTHON_APPEND_TYPE, DEFAULT_APPEND_TYPE).toUpperCase());
                    switch (pathAppendValue) {
                        case BEFORE:
                            for (File cacheDir : packages) {
                                sb.append(cacheDir);
                                sb.append(java.io.File.pathSeparator);
                            }

                            sb.append(path);

                            log.info("Prepending javacpp python path: {}", sb.toString());
                            break;
                        case AFTER:
                            sb.append(path);

                            for (File cacheDir : packages) {
                                sb.append(cacheDir);
                                sb.append(java.io.File.pathSeparator);
                            }

                            log.info("Appending javacpp python path " + sb.toString());
                            break;
                        case NONE:
                            log.info("Not appending javacpp path");
                            sb.append(path);
                            break;
                    }

                    //prepend the javacpp packages
                    log.info("Final python path: {}", sb.toString());

                    Py_SetPath(sb.toString());
                }
            } catch (IOException e) {
                log.error("Failed to set python path.", e);
            }
        } else {
            throw new IllegalStateException("Unable to reset python path. Already initialized.");
        }
    }
    
}
