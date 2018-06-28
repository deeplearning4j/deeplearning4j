package org.nd4j.tensorflow.conversion;

import org.bytedeco.javacpp.tensorflow;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

public class TensorflowDeAllocatorHolder {

    private   static List<tensorflow.Deallocator_Pointer_long_Pointer> calling = new CopyOnWriteArrayList<>();

    public static void addDeAllocatorRef(tensorflow.Deallocator_Pointer_long_Pointer pointer) {
        calling.add(pointer);
    }

    public static  List<tensorflow.Deallocator_Pointer_long_Pointer> currentList() {
        return calling;
    }


}
