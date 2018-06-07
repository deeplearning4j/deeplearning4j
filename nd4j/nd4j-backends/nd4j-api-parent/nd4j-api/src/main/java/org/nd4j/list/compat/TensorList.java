package org.nd4j.list.compat;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;

/**
 * This class implements basic storage for NDArrays
 */
@Slf4j
public class TensorList {
    private final String name;
    private ArrayList<INDArray> list = new ArrayList<>();

    public TensorList(@NonNull String name) {
        this.name = name;
    }

    public TensorList(@NonNull String name, @NonNull INDArray source) {
        this.name = name;
    }

    public INDArray get(int index) {
        return list.get(index);
    }

    public void put(int index, @NonNull INDArray array) {
        // TODO: if we want to validate shape - we should do it here

        list.ensureCapacity(index + 1);
        list.add(index, array);
    }

    public INDArray stack() {
        return Nd4j.pile(list);
    }

    public int size() {
        return list.size();
    }

    public String getName() {
        return name;
    }
}
