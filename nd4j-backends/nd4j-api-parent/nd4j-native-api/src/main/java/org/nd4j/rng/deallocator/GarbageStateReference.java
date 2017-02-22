package org.nd4j.rng.deallocator;

import lombok.Getter;
import org.bytedeco.javacpp.Pointer;

import java.lang.ref.ReferenceQueue;
import java.lang.ref.WeakReference;

/**
 * Weak reference for NativeRandom garbage collector
 *
 * @author raver119@gmail.com
 */
public class GarbageStateReference extends WeakReference<NativePack> {
    @Getter
    private Pointer statePointer;

    public GarbageStateReference(NativePack referent, ReferenceQueue<? super NativePack> queue) {
        super(referent, queue);
        this.statePointer = referent.getStatePointer();
        if (this.statePointer == null)
            throw new IllegalStateException("statePointer shouldn't be NULL");
    }
}
