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
import org.bytedeco.javacpp.Pointer;

import java.io.Closeable;
import java.util.HashSet;
import java.util.Set;

import static org.bytedeco.cpython.global.python.*;

public class PythonGC implements Closeable {

    private PythonGC previousFrame = null;
    private boolean active = true;
    private static PythonGC currentFrame = new PythonGC();

    private Set<PyObject> objects = new HashSet<>();

    private boolean alreadyRegistered(PyObject pyObject) {
        if (objects.contains(pyObject)) {
            return true;
        }
        if (previousFrame == null) {
            return false;
        }
        return previousFrame.alreadyRegistered(pyObject);

    }

    private void addObject(PythonObject pythonObject) {
        if (!active) return;
        if (Pointer.isNull(pythonObject.getNativePythonObject()))return;
        if (alreadyRegistered(pythonObject.getNativePythonObject())) {
            return;
        }
        objects.add(pythonObject.getNativePythonObject());
    }

    public static void register(PythonObject pythonObject) {
        currentFrame.addObject(pythonObject);
    }

    public static void keep(PythonObject pythonObject) {
        currentFrame.objects.remove(pythonObject.getNativePythonObject());
        if (currentFrame.previousFrame != null) {
            currentFrame.previousFrame.addObject(pythonObject);
        }
    }

    private PythonGC() {
    }

    public static PythonGC watch() {
        PythonGC ret = new PythonGC();
        ret.previousFrame = currentFrame;
        ret.active = currentFrame.active;
        currentFrame = ret;
        return ret;
    }

    private void collect() {
        for (PyObject pyObject : objects) {
            // TODO find out how globals gets collected here
            if (pyObject.equals(Python.globals().getNativePythonObject())) continue;
//            try{
//                System.out.println(PythonTypes.STR.toJava(new PythonObject(pyObject, false)));
//            }catch (Exception e){}
            Py_DecRef(pyObject);

        }
        this.objects = new HashSet<>();
    }

    @Override
    public void close() {
        if (active) collect();
        currentFrame = previousFrame;
    }

    public static boolean isWatching() {
        if (!currentFrame.active) return false;
        return currentFrame.previousFrame != null;
    }

    public static PythonGC pause() {
        PythonGC pausedFrame = new PythonGC();
        pausedFrame.active = false;
        pausedFrame.previousFrame = currentFrame;
        currentFrame = pausedFrame;
        return pausedFrame;
    }

    public static void resume() {
        if (currentFrame.active) {
            throw new RuntimeException("GC not paused!");
        }
        currentFrame = currentFrame.previousFrame;
    }
}
