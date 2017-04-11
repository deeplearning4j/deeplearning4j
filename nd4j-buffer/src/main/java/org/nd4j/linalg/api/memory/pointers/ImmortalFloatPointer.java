package org.nd4j.linalg.api.memory.pointers;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class ImmortalFloatPointer extends FloatPointer {
    private Pointer pointer;

    public ImmortalFloatPointer(PagedPointer pointer) {
        this.pointer = pointer;

        this.address = pointer.address();
        this.capacity = pointer.capacity();
        this.limit = pointer.limit();
        this.position = 0;
    }

    @Override
    public void deallocate() {
        log.info("deallocate 1");
        //super.deallocate();
    }

    @Override
    public void deallocate(boolean deallocate) {
        log.info("deallocate 2");
      //  super.deallocate(deallocate);
    }
}
