package org.deeplearning4j.nn.api.activations;

import org.deeplearning4j.nn.api.MaskState;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Activations {

    int size();

    INDArray get(int idx);

    INDArray getMask(int idx);

    MaskState getMaskState(int idx);

    void set(int idx, INDArray activations);

    void setMask(int idx, INDArray mask);

    void setMaskState(int idx, MaskState maskState);

    void clear();


    INDArray[] getAsArray();

    INDArray[] getMaskAsArray();

    MaskState[] getMaskStateAsArray();

    void setFromArray(INDArray[] activations);

    void setMaskFromArray(INDArray[] mask, MaskState[] maskStates);

    /**
     *
     * @param workspaceId
     * @return              This, after leveraging any attached arrays (inc. masks) to the specified workspace
     */
    Activations leverageTo(String workspaceId);

    /**
     *
     * @return              This, after migrating any attached arrays (inc. masks) to the current workspace
     */
    Activations migrate();

}
