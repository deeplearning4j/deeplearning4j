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

/**
 *
 * This class helps control the runtime's {@link PythonExecutioner} -
 * the {@link PythonExecutioner} is heavily system properties based.
 * Various aspects of the python executioner can be controlled with
 * the properties in this class. Python's core behavior of initialization,
 * python path setting, and working with javacpp's embedded cpython
 * are keys to integrating the python executioner successfully with various applications.
 *
 * @author Adam Gibson
 */
public class PythonConstants {
    public final static String DEFAULT_PYTHON_PATH_PROPERTY = "org.eclipse.python4j.path";
    public final static String JAVACPP_PYTHON_APPEND_TYPE = "org.eclipse.python4j.path.append";
    //for embedded execution, this is to ensure we allow customization of the gil state releasing when running in another embedded python situation
    public final static String RELEASE_GIL_AUTOMATICALLY = "org.eclipse.python4j.release_gil_automatically";
    public final static String DEFAULT_RELEASE_GIL_AUTOMATICALLY = "true";
    public final static String DEFAULT_APPEND_TYPE = "before";
    public final static String INITIALIZE_PYTHON = "org.eclipse.python4j.python.initialize";
    public final static String DEFAULT_INITIALIZE_PYTHON = "true";
    public final static String PYTHON_EXEC_RESOURCE = "org/nd4j/python4j/pythonexec/pythonexec.py";
    final static String PYTHON_EXCEPTION_KEY = "__python_exception__";
    public final static String CREATE_NPY_VIA_PYTHON = "org.eclipse.python4j.create_npy_python";
    public final static String DEFAULT_CREATE_NPY_VIA_PYTHON = "false";

    /**
     * Controls how to create the numpy array objects associated
     * with the NumpyArray.java module.
     *
     * Depending on how threading is handled, Py_Type() causes a JVM crash
     * when used. Py_Type() is used to obtain the type of a numpy array.
     * Defaults to false, as most of the time this is less performant and not needed.
     *
     * The python based method uses raw pointer address + ctypes inline to create the proper numpy array
     * on the python side.
     * Otherwise, a more direct c based approach is used.
     * @return
     */
   public static boolean createNpyViaPython() {
       return Boolean.parseBoolean(System.getProperty(CREATE_NPY_VIA_PYTHON,DEFAULT_CREATE_NPY_VIA_PYTHON));
   }

    /**
     * Setter for the associated property
     * from {@link #createNpyViaPython()}
     * please see this function for more information.
     * @param createNpyViaPython
     */
   public static void setCreateNpyViaPython(boolean createNpyViaPython) {
       System.setProperty(CREATE_NPY_VIA_PYTHON,String.valueOf(createNpyViaPython));
   }


    /**
     * Sets the default python path.
     * See {@link #defaultPythonPath()}
     * for more information.
     * @param newPythonPath the new python path to use
     */
    public static void setDefaultPythonPath(String newPythonPath) {
        System.setProperty(DEFAULT_PYTHON_PATH_PROPERTY,newPythonPath);
    }

    /**
     * Returns the default python path.
     * This python path should be initialized before the {@link PythonExecutioner}
     * is called.
     * @return
     */
    public static String defaultPythonPath() {
        return System.getProperty(PythonConstants.DEFAULT_PYTHON_PATH_PROPERTY);
    }

    /**
     * Returns whether to initialize python or not.
     * This property is used when python should be initialized manually.
     * Normally, the {@link PythonExecutioner} will handle initialization
     * in its {@link PythonExecutioner#init()} method
     *
     * @return
     */
    public static boolean initializePython() {
        return Boolean.parseBoolean(System.getProperty(INITIALIZE_PYTHON,DEFAULT_INITIALIZE_PYTHON));
    }

    /**
     * See {@link #initializePython()}
     *  for more information on this property.
     *  This is the setter method for the associated value.
     * @param initializePython whether to initialize python or not
     */
    public static void setInitializePython(boolean initializePython) {
        System.setProperty(INITIALIZE_PYTHON,String.valueOf(initializePython));
    }


    /**
     * Returns the default javacpp python append type.
     * In javacpp's cython module, it comes with built in support
     * for determining the python path of most modules.
     *
     * This can clash when invoking python using another distribution of python
     * such as anaconda. This property allows the user to control how javacpp
     * interacts with a different python present on the classpath.
     *
     * The default value is {@link #DEFAULT_APPEND_TYPE}
     * @return
     */
    public static PythonExecutioner.JavaCppPathType javaCppPythonAppendType() {
        return PythonExecutioner.JavaCppPathType.valueOf(System.getProperty(JAVACPP_PYTHON_APPEND_TYPE,DEFAULT_APPEND_TYPE).toUpperCase());
    }

    /**
     * Setter for the javacpp append type.
     * See {@link #javaCppPythonAppendType()}
     * for more information on value set by this setter.
     * @param appendType the append type to use
     */
    public static void setJavacppPythonAppendType(PythonExecutioner.JavaCppPathType appendType) {
        System.setProperty(JAVACPP_PYTHON_APPEND_TYPE,appendType.name());
    }


    /**
     * See {@link #releaseGilAutomatically()}
     * for more information on this setter.
     * @param releaseGilAutomatically whether to release the gil automatically or not.
     */
    public static void setReleaseGilAutomatically(boolean releaseGilAutomatically) {
        System.setProperty(RELEASE_GIL_AUTOMATICALLY,String.valueOf(releaseGilAutomatically));
    }

    /**
     * Returns true if the GIL is released automatically or not.
     * For linking against applications where python is already present
     * this is a knob allowing people to turn automatic python thread management off.
     * This is enabled by default. See {@link #RELEASE_GIL_AUTOMATICALLY}
     * and its default value {@link #DEFAULT_RELEASE_GIL_AUTOMATICALLY}
     * @return
     */
    public final static boolean releaseGilAutomatically() {
        return Boolean.parseBoolean(System.getProperty(RELEASE_GIL_AUTOMATICALLY,DEFAULT_RELEASE_GIL_AUTOMATICALLY));
    }

}
