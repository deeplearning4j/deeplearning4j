package org.deeplearning4j.nn.api.activations;

import org.deeplearning4j.nn.api.MaskState;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Activations {

    int size();

    INDArray get(int idx);

    INDArray getMask(int idx);

    MaskState getMaskState(int idx);

    void set(int idx, INDArray activations);

    void set(int idx, INDArray activations, INDArray mask, MaskState maskState);

    void setMask(int idx, INDArray mask);

    void setMask(int idx, INDArray mask, MaskState maskState);

    void setMaskState(int idx, MaskState maskState);

    void clear();

    boolean anyActivationsNull();

    boolean anyMasksNull();


    INDArray[] getAsArray();

    INDArray[] getMaskAsArray();

    MaskState[] getMaskStateAsArray();

    void setFromArray(INDArray[] activations);

    void setMaskFromArray(INDArray[] mask, MaskState[] maskStates);

    Activations cloneShallow();

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

    /**
     * @return These activations (same object) after detaching activations + masks from workspace
     */
    Activations detach();


    /**
     * Get a subset of the activations (inc. masks). This may return the original object in some cases
     * @param idx
     * @return
     */
    Activations getSubset(int idx);

}
