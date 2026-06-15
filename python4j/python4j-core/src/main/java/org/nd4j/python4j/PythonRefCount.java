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
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;

import static org.bytedeco.cpython.global.python.PyUnstable_IsImmortal;
import static org.bytedeco.cpython.global.python.Py_TYPE;

/**
 * Replacement for Py_IncRef/Py_DecRef which were removed from the cpython JavaCPP bindings
 * in 3.14.3-1.5.13 because CPython 3.14 made them inline functions that cannot be exported
 * as shared library symbols.
 *
 * This utility directly manipulates the ob_refcnt field at offset 0 of the PyObject struct
 * using JavaCPP's LongPointer, and calls tp_dealloc when the reference count reaches zero.
 *
 * Immortal objects (None, True, False, small ints, etc.) are detected via
 * PyUnstable_IsImmortal and their refcnts are never modified.
 */
public class PythonRefCount {

    private PythonRefCount() {
        // utility class
    }

    /**
     * Increment the reference count of a Python object.
     * Equivalent to Py_IncRef / Py_XINCREF (null-safe).
     *
     * @param obj the PyObject to increment, may be null
     */
    public static void incRef(PyObject obj) {
        if (obj == null || Pointer.isNull(obj)) {
            return;
        }
        if (PyUnstable_IsImmortal(obj) != 0) {
            return;
        }
        LongPointer refcntPtr = new LongPointer(obj);
        long refcnt = refcntPtr.get(0);
        refcntPtr.put(0, refcnt + 1);
    }

    /**
     * Decrement the reference count of a Python object.
     * When the count reaches zero, the object's tp_dealloc is called.
     * Equivalent to Py_DecRef / Py_XDECREF (null-safe).
     *
     * @param obj the PyObject to decrement, may be null
     */
    public static void decRef(PyObject obj) {
        if (obj == null || Pointer.isNull(obj)) {
            return;
        }
        if (PyUnstable_IsImmortal(obj) != 0) {
            return;
        }
        LongPointer refcntPtr = new LongPointer(obj);
        long refcnt = refcntPtr.get(0);
        if (refcnt <= 1) {
            // Reference count is about to hit zero - deallocate the object
            refcntPtr.put(0, 0);
            Py_TYPE(obj).tp_dealloc().call(obj);
        } else {
            refcntPtr.put(0, refcnt - 1);
        }
    }
}
