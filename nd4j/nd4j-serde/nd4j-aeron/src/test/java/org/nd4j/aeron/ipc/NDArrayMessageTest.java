/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.aeron.ipc;

import org.agrona.DirectBuffer;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.Assert.assertEquals;

/**
 * Created by agibsonccc on 11/6/16.
 */
public class NDArrayMessageTest {

    @Test
    public void testNDArrayMessageToAndFrom() {
        NDArrayMessage message = NDArrayMessage.wholeArrayUpdate(Nd4j.scalar(1.0));
        DirectBuffer bufferConvert = NDArrayMessage.toBuffer(message);
        bufferConvert.byteBuffer().rewind();
        NDArrayMessage newMessage = NDArrayMessage.fromBuffer(bufferConvert, 0);
        assertEquals(message, newMessage);

        INDArray compressed = Nd4j.getCompressor().compress(Nd4j.scalar(1.0), "GZIP");
        NDArrayMessage messageCompressed = NDArrayMessage.wholeArrayUpdate(compressed);
        DirectBuffer bufferConvertCompressed = NDArrayMessage.toBuffer(messageCompressed);
        NDArrayMessage newMessageTest = NDArrayMessage.fromBuffer(bufferConvertCompressed, 0);
        assertEquals(messageCompressed, newMessageTest);


    }


}
