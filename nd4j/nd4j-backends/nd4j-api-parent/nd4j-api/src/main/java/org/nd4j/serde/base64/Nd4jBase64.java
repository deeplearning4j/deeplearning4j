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

package org.nd4j.serde.base64;

import org.apache.commons.net.util.Base64;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

/**
 * NDArray as base 64
 *
 * @author Adam Gibson
 */
public class Nd4jBase64 {

    private Nd4jBase64() {}

    /**
     * Returns an ndarray
     * as base 64
     * @param arr the array to write
     * @return the base 64 representation of the binary
     * ndarray
     */
    public static String base64String(INDArray arr) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        DataOutputStream dos = new DataOutputStream(bos);
        Nd4j.write(arr, dos);
        return Base64.encodeBase64String(bos.toByteArray());
    }

    /**
     * Create an ndarray from a base 64
     * representation
     * @param base64 the base 64 to convert
     * @return the ndarray from base 64
     */
    public static INDArray fromBase64(String base64) {
        byte[] arr = Base64.decodeBase64(base64);
        ByteArrayInputStream bis = new ByteArrayInputStream(arr);
        DataInputStream dis = new DataInputStream(bis);
        return Nd4j.read(dis);
    }
}
