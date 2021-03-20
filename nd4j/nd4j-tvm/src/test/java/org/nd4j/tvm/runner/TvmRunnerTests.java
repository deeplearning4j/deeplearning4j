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
package org.nd4j.tvm.runner;

import org.bytedeco.cpython.*;


import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.numpy.global.numpy.*;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.io.TempDir;

@Disabled
@Tag(TagNames.FILE_IO)
@Tag(TagNames.DIST_SYSTEMS)
@NativeTag
public class TvmRunnerTests {

    static void PrepareTestLibs(String libPath) throws Exception {
        Py_AddPath(org.bytedeco.tvm.presets.tvm.cachePackages());
        Py_Initialize();
        if (_import_array() < 0) {
            System.err.println("numpy.core.multiarray failed to import");
            PyErr_Print();
            System.exit(-1);
        }
        PyObject globals = PyModule_GetDict(PyImport_AddModule("__main__"));

        PyRun_StringFlags("\"\"\"Script to prepare test_relay_add.so\"\"\"\n"
                + "import tvm\n"
                + "import numpy as np\n"
                + "from tvm import relay\n"
                + "import os\n"

                + "x = relay.var(\"x\", shape=(1, 1), dtype=\"float32\")\n"
                + "y = relay.var(\"y\", shape=(1, 1), dtype=\"float32\")\n"
                + "params = {\"y\": np.ones((1, 1), dtype=\"float32\")}\n"
                + "mod = tvm.IRModule.from_expr(relay.Function([x, y], x + y))\n"
                + "# build a module\n"
                + "compiled_lib = relay.build(mod, tvm.target.create(\"llvm\"), params=params)\n"
                + "# export it as a shared library\n"
                + "dylib_path = os.path.join(\"" + libPath + "\", \"test_relay_add.so\")\n"
                + "compiled_lib.export_library(dylib_path)\n",

                Py_file_input, globals, globals, null);

        if (PyErr_Occurred() != null) {
            System.err.println("Python error occurred");
            PyErr_Print();
            System.exit(-1);
        }
    }

    @Test
    public void testAdd(@TempDir Path tempDir) throws Exception {
        /* try to use MKL when available */
        System.setProperty("org.bytedeco.openblas.load", "mkl");

        File libPath = tempDir.resolve("lib").toFile();
        PrepareTestLibs(libPath.getAbsolutePath().replace(File.separatorChar, '/'));
        File f = new File(libPath, "test_relay_add.so");
        INDArray x = Nd4j.scalar(1.0f).reshape(1,1);
        TvmRunner tvmRunner = TvmRunner.builder()
                .modelUri(f.getAbsolutePath())
                .build();
        Map<String,INDArray> inputs = new LinkedHashMap<>();
        inputs.put("x",x);
        Map<String, INDArray> exec = tvmRunner.exec(inputs);
        INDArray z = exec.get("0");
        assertEquals(2.0,z.sumNumber().doubleValue(),1e-1);
    }
}
