package org.nd4j.parameterserver.parameteraveraging;

import io.aeron.Aeron;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.aeron.ipc.AeronNDArrayPublisher;
import org.nd4j.aeron.ipc.NDArrayCallback;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Publishing listener for
 * publishing to a master url.
 *
 * @author Adam Gibson
 */
@Data
@AllArgsConstructor
@Slf4j
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
        try (AeronNDArrayPublisher publisher =   AeronNDArrayPublisher.builder()
                .streamId(streamId)
                .ctx(aeronContext).channel(masterUrl)
                .build()) {
            publisher.publish(arr);
            log.debug("NDArray PublishingListener publishing to channel " + masterUrl + ":" + streamId);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }
}
