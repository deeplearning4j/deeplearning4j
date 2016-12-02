package org.nd4j.parameterserver.updater;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.HashMap;
import java.util.Map;

/**
 * Adds the 2 arrays together,
 * synchronizing when
 * all updates have been collected.
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class SynchronousParameterUpdater implements ParameterServerUpdater {

    private int workers = Runtime.getRuntime().availableProcessors();
    private int accumulatedUpdates;
    private static ObjectMapper objectMapper = new ObjectMapper();
    public SynchronousParameterUpdater(int workers) {
        this.workers = workers;
    }

    /**
     * Returns the current status of this parameter server
     * updater
     *
     * @return
     */
    @Override
    public Map<String, Number> status() {
        Map<String,Number> ret = new HashMap<>();
        ret.put("workers",workers);
        ret.put("accumulatedUpdates",accumulatedUpdates);
        return ret;
    }

    /**
     * Serialize this updater as json
     *
     * @return
     */
    @Override
    public String toJson() {
        try {
            return objectMapper.writeValueAsString(status());
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Reset internal counters
     * such as number of updates accumulated.
     */
    @Override
    public void reset() {
        accumulatedUpdates = 0;
    }

    /**
     * Returns true if
     * the updater has accumulated enough ndarrays to
     * replicate to the workers
     *
     * @return true if replication should happen,false otherwise
     */
    @Override
    public boolean shouldReplicate() {
        return accumulatedUpdates == workers;
    }

    /**
     * Updates result
     * based on arr
     *
     * @param arr    the array to update
     * @param result the result ndarray to update
     */
    @Override
    public void update(INDArray arr, INDArray result) {
        result.addi(arr);
        accumulatedUpdates++;
    }
}
