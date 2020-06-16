/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

import org.eclipse.python4j.Python;
import org.eclipse.python4j.PythonGC;
import org.eclipse.python4j.PythonObject;
import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import javax.annotation.concurrent.NotThreadSafe;


@NotThreadSafe
public class PythonNumpyGCTest {

    @Test
    public void testGC(){
        PythonObject gcModule = Python.importModule("gc");
        PythonObject getObjects = gcModule.attr("get_objects");
        PythonObject pyObjCount1 = Python.len(getObjects.call());
        long objCount1 =  pyObjCount1.toLong();
        PythonObject pyList = Python.list();
        pyList.attr("append").call(new PythonObject(Nd4j.linspace(1, 10, 10)));
        pyList.attr("append").call(1.0);
        pyList.attr("append").call(true);
        PythonObject pyObjCount2 = Python.len(getObjects.call());
        long objCount2 =  pyObjCount2.toLong();
        long diff = objCount2 - objCount1;
        Assert.assertTrue(diff > 2);
        try(PythonGC gc = PythonGC.watch()){
            PythonObject pyList2 = Python.list();
            pyList2.attr("append").call(new PythonObject(Nd4j.linspace(1, 10, 10)));
            pyList2.attr("append").call(1.0);
            pyList2.attr("append").call(true);
        }
        PythonObject pyObjCount3 = Python.len(getObjects.call());
        long objCount3 =  pyObjCount3.toLong();
        diff = objCount3 - objCount2;
        Assert.assertTrue(diff <= 2);// 2 objects created during function call
    }
}
