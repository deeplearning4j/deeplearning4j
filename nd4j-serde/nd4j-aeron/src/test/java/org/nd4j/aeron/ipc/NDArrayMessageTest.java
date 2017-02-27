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
