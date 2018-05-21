package org.nd4j.parameterserver;

import io.aeron.Aeron;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.aeron.ipc.AeronNDArrayPublisher;
import org.nd4j.aeron.ipc.NDArrayCallback;
import org.nd4j.aeron.ipc.NDArrayMessage;
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
     * A listener for ndarray message
     *
     * @param message the message for the callback
     */
    @Override
    public void onNDArrayMessage(NDArrayMessage message) {
        try (AeronNDArrayPublisher publisher = AeronNDArrayPublisher.builder().streamId(streamId).ctx(aeronContext)
                        .channel(masterUrl).build()) {
            publisher.publish(message);
            log.debug("NDArray PublishingListener publishing to channel " + masterUrl + ":" + streamId);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

    }

    /**
     * Used for partial updates using tensor along
     * dimension
     *
     * @param arr        the array to count as an update
     * @param idx        the index for the tensor along dimension
     * @param dimensions the dimensions to act on for the tensor along dimension
     */
    @Override
    public void onNDArrayPartial(INDArray arr, long idx, int... dimensions) {
        try (AeronNDArrayPublisher publisher = AeronNDArrayPublisher.builder().streamId(streamId).ctx(aeronContext)
                        .channel(masterUrl).build()) {
            publisher.publish(NDArrayMessage.builder().arr(arr).dimensions(dimensions).index(idx).build());
            log.debug("NDArray PublishingListener publishing to channel " + masterUrl + ":" + streamId);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Setup an ndarray
     *
     * @param arr
     */
    @Override
    public void onNDArray(INDArray arr) {
        try (AeronNDArrayPublisher publisher = AeronNDArrayPublisher.builder().streamId(streamId).ctx(aeronContext)
                        .channel(masterUrl).build()) {
            publisher.publish(arr);
            log.debug("NDArray PublishingListener publishing to channel " + masterUrl + ":" + streamId);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }


    }
}
