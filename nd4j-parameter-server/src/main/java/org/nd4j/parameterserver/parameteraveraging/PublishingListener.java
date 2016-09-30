package org.nd4j.parameterserver.parameteraveraging;

import io.aeron.Aeron;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.nd4j.aeron.ipc.AeronNDArrayPublisher;
import org.nd4j.aeron.ipc.NDArrayCallback;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Publishing listener for publishing to a master url.
 *
 * @author Adam Gibson
 */
@Data
@AllArgsConstructor
public class PublishingListener implements NDArrayCallback {
    private String masterUrl;
    private int streamId;
    private Aeron.Context aeronContext;
    /**
     * Setup an ndarray
     *
     * @param arr
     */
    @Override
    public void onNDArray(INDArray arr) {
        AeronNDArrayPublisher publisher =   AeronNDArrayPublisher.builder().streamId(10)
                .ctx(aeronContext).channel(masterUrl)
                .build();
        try {
            publisher.publish(arr);
            publisher.close();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }
}
