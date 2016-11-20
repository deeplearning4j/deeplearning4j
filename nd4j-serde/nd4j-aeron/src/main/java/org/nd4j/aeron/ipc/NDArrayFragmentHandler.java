package org.nd4j.aeron.ipc;

import io.aeron.logbuffer.FragmentHandler;
import io.aeron.logbuffer.Header;
import org.agrona.DirectBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;



/**
 * NDArray fragment handler
 * for listening to an aeron queue
 *
 * @author Adam Gibson
 */
public class NDArrayFragmentHandler implements FragmentHandler {
    private NDArrayCallback ndArrayCallback;

    public NDArrayFragmentHandler(NDArrayCallback ndArrayCallback) {
        this.ndArrayCallback = ndArrayCallback;
    }

    /**
     * Callback for handling
     * fragments of data being read from a log.
     *
     * @param buffer containing the data.
     * @param offset at which the data begins.
     * @param length of the data in bytes.
     * @param header representing the meta data for the data.
     */
    @Override
    public void onFragment(DirectBuffer buffer, int offset, int length, Header header) {
        NDArrayMessage message = NDArrayMessage.fromBuffer(buffer,offset);
        INDArray arr = message.getArr();
        //of note for ndarrays
        int[] dimensions = message.getDimensions();
        boolean whole = dimensions.length == 1 && dimensions[0] == -1;

        if(!whole)
            ndArrayCallback.onNDArrayPartial(arr,message.getIndex(),dimensions);
        else
            ndArrayCallback.onNDArray(arr);

    }
}
