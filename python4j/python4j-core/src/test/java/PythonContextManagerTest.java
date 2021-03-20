
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


import org.junit.jupiter.api.Tag;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.python4j.Python;
import org.nd4j.python4j.PythonContextManager;
import org.nd4j.python4j.PythonExecutioner;

import org.junit.jupiter.api.Test;
import org.nd4j.python4j.PythonGIL;

import javax.annotation.concurrent.NotThreadSafe;

import static org.junit.jupiter.api.Assertions.assertEquals;

@NotThreadSafe
@Tag(TagNames.FILE_IO)
@NativeTag
@Tag(TagNames.FILE_IO)
@Tag(TagNames.PYTHON)
public class PythonContextManagerTest {

    @Test
    public void testInt() throws Exception {
        try(PythonGIL pythonGIL = PythonGIL.lock()) {
            Python.setContext("context1");
            Python.exec("a = 1");
            Python.setContext("context2");
            Python.exec("a = 2");
            Python.setContext("context3");
            Python.exec("a = 3");


            Python.setContext("context1");
            assertEquals(1, PythonExecutioner.getVariable("a").toInt());

            Python.setContext("context2");
            assertEquals(2, PythonExecutioner.getVariable("a").toInt());

            Python.setContext("context3");
            assertEquals(3, PythonExecutioner.getVariable("a").toInt());

            PythonContextManager.deleteNonMainContexts();

        }

    }

}
