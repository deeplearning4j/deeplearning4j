/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */


package org.datavec.python;

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import javax.annotation.Nonnull;


@Data
@NoArgsConstructor
/**
 * PythonJob is the right abstraction for executing multiple python scripts
 * in a multi thread stateful environment. The setup-and-run mode allows your
 * "setup" code (imports, model loading etc) to be executed only once.
 */
public class PythonJob {

    private String code;
    private String name;
    private String context;
    private boolean setupRunMode;
    private PythonObject runF;

    static {
        new PythonExecutioner();
    }

    @Builder
    /**
     * @param name Name for the python job.
     * @param code Python code.
     * @param setupRunMode If true, the python code is expected to have two methods: setup(), which takes no arguments,
     *                     and run() which takes some or no arguments. setup() method is executed once,
     *                     and the run() method is called with the inputs(if any) per transaction, and is expected to return a dictionary
     *                     mapping from output variable names (str) to output values.
     *                     If false, the full script is run on each transaction and the output variables are obtained from the global namespace
     *                     after execution.
     */
    public PythonJob(@Nonnull String name, @Nonnull String code, boolean setupRunMode) throws Exception {
        this.name = name;
        this.code = code;
        this.setupRunMode = setupRunMode;
        context = "__job_" + name;
        if (PythonContextManager.hasContext(context)) {
            throw new PythonException("Unable to create python job " + name + ". Context " + context + " already exists!");
        }
        if (setupRunMode) setup();
    }


    /**
     * Clears all variables in current context and calls setup()
     */
    public void clearState() throws Exception {
        String context = this.context;
        PythonContextManager.setContext("main");
        PythonContextManager.deleteContext(context);
        this.context = context;
        setup();
    }

    public void setup() throws Exception {
        try (PythonGIL gil = PythonGIL.lock()) {
            PythonContextManager.setContext(context);
            PythonObject runF = PythonExecutioner.getVariable("run");
            if (runF.isNone() || !Python.callable(runF)) {
                PythonExecutioner.exec(code);
                runF = PythonExecutioner.getVariable("run");
            }
            if (runF.isNone() || !Python.callable(runF)) {
                throw new PythonException("run() method not found! " +
                        "If a PythonJob is created with 'setup and run' " +
                        "mode enabled, the associated python code is " +
                        "expected to contain a run() method " +
                        "(with or without arguments).");
            }
            this.runF = runF;
            PythonObject setupF = PythonExecutioner.getVariable("setup");
            if (!setupF.isNone()) {
                setupF.call();
            }
        }
    }

    public void exec(PythonVariables inputs, PythonVariables outputs) throws Exception {
        try (PythonGIL gil = PythonGIL.lock()) {
            PythonContextManager.setContext(context);
            if (!setupRunMode) {
                PythonExecutioner.exec(code, inputs, outputs);
                return;
            }
            PythonExecutioner.setVariables(inputs);

            PythonObject inspect = Python.importModule("inspect");
            PythonObject getfullargspec = inspect.attr("getfullargspec");
            PythonObject argspec = getfullargspec.call(runF);
            PythonObject argsList = argspec.attr("args");
            PythonObject runargs = Python.dict();
            int argsCount = Python.len(argsList).toInt();
            for (int i = 0; i < argsCount; i++) {
                PythonObject arg = argsList.get(i);
                PythonObject val = Python.globals().get(arg);
                if (val.isNone()) {
                    throw new PythonException("Input value not received for run() argument: " + arg.toString());
                }
                runargs.set(arg, val);
            }
            PythonObject outDict = runF.callWithKwargs(runargs);
            Python.globals().attr("update").call(outDict);

            PythonExecutioner.getVariables(outputs);
            inspect.del();
            getfullargspec.del();
            argspec.del();
            runargs.del();
        }
    }

    public PythonVariables execAndReturnAllVariables(PythonVariables inputs) throws Exception {
        try (PythonGIL gil = PythonGIL.lock()) {
            PythonContextManager.setContext(context);
            if (!setupRunMode) {
                return PythonExecutioner.execAndReturnAllVariables(code, inputs);
            }
            PythonExecutioner.setVariables(inputs);
            PythonObject inspect = Python.importModule("inspect");
            PythonObject getfullargspec = inspect.attr("getfullargspec");
            PythonObject argspec = getfullargspec.call(runF);
            PythonObject argsList = argspec.attr("args");
            PythonObject runargs = Python.dict();
            int argsCount = Python.len(argsList).toInt();
            for (int i = 0; i < argsCount; i++) {
                PythonObject arg = argsList.get(i);
                PythonObject val = Python.globals().get(arg);
                if (val.isNone()) {
                    throw new PythonException("Input value not received for run() argument: " + arg.toString());
                }
                runargs.set(arg, val);
            }
            PythonObject outDict = runF.callWithKwargs(runargs);
            Python.globals().attr("update").call(outDict);
            inspect.del();
            getfullargspec.del();
            argspec.del();
            runargs.del();
            return PythonExecutioner.getAllVariables();
        }
    }


}
