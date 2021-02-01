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

package org.deeplearning4j.rl4j.mdp.gym;

import java.io.IOException;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.ActionSpace;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.HighLowDiscrete;
import org.deeplearning4j.rl4j.space.Encodable;
import org.deeplearning4j.rl4j.space.ObservationSpace;

import org.bytedeco.cpython.*;
import org.bytedeco.numpy.*;
import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.*;

/**
 * An MDP for OpenAI Gym: https://gym.openai.com/
 *
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/12/16.
 * @author saudet
 */
@Slf4j
public class GymEnv<OBSERVATION extends Encodable, A, AS extends ActionSpace<A>> implements MDP<OBSERVATION, A, AS> {

    public static final String GYM_MONITOR_DIR = "/tmp/gym-dqn";

    private static void checkPythonError() {
        if (PyErr_Occurred() != null) {
            PyErr_Print();
            throw new RuntimeException("Python error occurred");
        }
    }

    private static Pointer program;
    private static PyObject globals;
    static {
        try {
            Py_AddPath(org.bytedeco.gym.presets.gym.cachePackages());
            program = Py_DecodeLocale(GymEnv.class.getSimpleName(), null);
            Py_SetProgramName(program);
            Py_Initialize();
            PyEval_InitThreads();
            PySys_SetArgvEx(1, program, 0);
            if (_import_array() < 0) {
                PyErr_Print();
                throw new RuntimeException("numpy.core.multiarray failed to import");
            }
            globals = PyModule_GetDict(PyImport_AddModule("__main__"));
            PyEval_SaveThread(); // just to release the GIL
        } catch (IOException e) {
            PyMem_RawFree(program);
            throw new RuntimeException(e);
        }
    }
    private PyObject locals;

    final protected DiscreteSpace actionSpace;
    final protected ObservationSpace<OBSERVATION> observationSpace;
    @Getter
    final private String envId;
    @Getter
    final private boolean render;
    @Getter
    final private boolean monitor;
    private ActionTransformer actionTransformer = null;
    private boolean done = false;

    public GymEnv(String envId, boolean render, boolean monitor) {
        this(envId, render, monitor, (Integer)null);
    }
    public GymEnv(String envId, boolean render, boolean monitor, Integer seed) {
        this.envId = envId;
        this.render = render;
        this.monitor = monitor;

        int gstate = PyGILState_Ensure();
        try {
            locals = PyDict_New();

            Py_DecRef(PyRun_StringFlags("import gym; env = gym.make('" + envId + "')", Py_single_input, globals, locals, null));
            checkPythonError();
            if (monitor) {
                Py_DecRef(PyRun_StringFlags("env = gym.wrappers.Monitor(env, '" + GYM_MONITOR_DIR + "')", Py_single_input, globals, locals, null));
                checkPythonError();
            }
            if (seed != null) {
                Py_DecRef(PyRun_StringFlags("env.seed(" + seed + ")", Py_single_input, globals, locals, null));
                checkPythonError();
            }
            PyObject shapeTuple = PyRun_StringFlags("env.observation_space.shape", Py_eval_input, globals, locals, null);
            int[] shape = new int[(int)PyTuple_Size(shapeTuple)];
            for (int i = 0; i < shape.length; i++) {
                shape[i] = (int)PyLong_AsLong(PyTuple_GetItem(shapeTuple, i));
            }
            observationSpace = (ObservationSpace<OBSERVATION>) new ArrayObservationSpace<Box>(shape);
            Py_DecRef(shapeTuple);

            PyObject n = PyRun_StringFlags("env.action_space.n", Py_eval_input, globals, locals, null);
            actionSpace = new DiscreteSpace((int)PyLong_AsLong(n));
            Py_DecRef(n);
            checkPythonError();
        } finally {
            PyGILState_Release(gstate);
        }
    }

    public GymEnv(String envId, boolean render, boolean monitor, int[] actions) {
        this(envId, render, monitor, null, actions);
    }
    public GymEnv(String envId, boolean render, boolean monitor, Integer seed, int[] actions) {
        this(envId, render, monitor, seed);
        actionTransformer = new ActionTransformer((HighLowDiscrete) getActionSpace(), actions);
    }

    @Override
    public ObservationSpace<OBSERVATION> getObservationSpace() {
        return observationSpace;
    }

    @Override
    public AS getActionSpace() {
        if (actionTransformer == null)
            return (AS) actionSpace;
        else
            return (AS) actionTransformer;
    }

    @Override
    public StepReply<OBSERVATION> step(A action) {
        int gstate = PyGILState_Ensure();
        try {
            if (render) {
                Py_DecRef(PyRun_StringFlags("env.render()", Py_single_input, globals, locals, null));
                checkPythonError();
            }
            Py_DecRef(PyRun_StringFlags("state, reward, done, info = env.step(" + (Integer)action +")", Py_single_input, globals, locals, null));
            checkPythonError();

            PyArrayObject state = new PyArrayObject(PyDict_GetItemString(locals, "state"));
            DoublePointer stateData = new DoublePointer(PyArray_BYTES(state)).capacity(PyArray_Size(state));
            SizeTPointer stateDims = PyArray_DIMS(state).capacity(PyArray_NDIM(state));

            double reward = PyFloat_AsDouble(PyDict_GetItemString(locals, "reward"));
            done = PyLong_AsLong(PyDict_GetItemString(locals, "done")) != 0;
            checkPythonError();

            double[] data = new double[(int)stateData.capacity()];
            stateData.get(data);

            return new StepReply(new Box(data), reward, done, null);
        } finally {
            PyGILState_Release(gstate);
        }
    }

    @Override
    public boolean isDone() {
        return done;
    }

    @Override
    public OBSERVATION reset() {
        int gstate = PyGILState_Ensure();
        try {
            Py_DecRef(PyRun_StringFlags("state = env.reset()", Py_single_input, globals, locals, null));
            checkPythonError();

            PyArrayObject state = new PyArrayObject(PyDict_GetItemString(locals, "state"));
            DoublePointer stateData = new DoublePointer(PyArray_BYTES(state)).capacity(PyArray_Size(state));
            SizeTPointer stateDims = PyArray_DIMS(state).capacity(PyArray_NDIM(state));
            checkPythonError();

            done = false;

            double[] data = new double[(int)stateData.capacity()];
            stateData.get(data);
            return (OBSERVATION) new Box(data);
        } finally {
            PyGILState_Release(gstate);
        }
    }

    @Override
    public void close() {
        int gstate = PyGILState_Ensure();
        try {
            Py_DecRef(PyRun_StringFlags("env.close()", Py_single_input, globals, locals, null));
            checkPythonError();
            Py_DecRef(locals);
        } finally {
            PyGILState_Release(gstate);
        }
    }

    @Override
    public GymEnv<OBSERVATION, A, AS> newInstance() {
        return new GymEnv<OBSERVATION, A, AS>(envId, render, monitor);
    }
}
