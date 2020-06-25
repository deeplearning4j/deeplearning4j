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


import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.python4j.NumpyArray;
import org.nd4j.python4j.PythonTypes;

import javax.annotation.concurrent.NotThreadSafe;

@NotThreadSafe
public class PythonNumpyServiceLoaderTest {

    @Test
    public void testServiceLoader(){
        Assert.assertEquals(NumpyArray.INSTANCE, PythonTypes.<INDArray>get("numpy.ndarray"));
        Assert.assertEquals(NumpyArray.INSTANCE, PythonTypes.getPythonTypeForJavaObject(Nd4j.zeros(1)));
    }
}
