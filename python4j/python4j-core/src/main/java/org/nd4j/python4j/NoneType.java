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

import javax.annotation.concurrent.NotThreadSafe;

/**
 * Based on work from:
 * https://github.com/invesdwin/invesdwin-context-python/blob/master/invesdwin-context-python-parent/invesdwin-context-python-runtime-python4j/src/main/java/de/invesdwin/context/python/runtime/python4j/internal/NonePythonType.java
 * Permission granted from original author here: https://github.com/eclipse/deeplearning4j/issues/9595
 *
 * @author Adam Gibson
 */
@NotThreadSafe
public class NoneType extends PythonType<Void> {

    private static final String ANS = "__ans__";
    private static final String ANS_EQUALS = ANS + " = ";


    public NoneType() {
        super("NoneType", Void.class);
    }

    @Override
    public boolean accepts(final Object javaObject) {
        return javaObject == null;
    }

    @Override
    public Void toJava(final PythonObject pythonObject) {
        return null;
    }

    @Override
    public PythonObject toPython(final Void javaObject) {
        return UncheckedPythonInterpreter.getInstance().newNone();
    }

}