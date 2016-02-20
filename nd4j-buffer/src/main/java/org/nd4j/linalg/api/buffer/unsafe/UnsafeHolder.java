package org.nd4j.linalg.api.buffer.unsafe;

import sun.misc.Unsafe;

import java.lang.reflect.Field;

/**
 * Unsafe singleton holder
 * @author Adam Gibson
 */
public class UnsafeHolder {
    private static Unsafe INSTANCE;


    private UnsafeHolder() {}

    /**
     * Unsafe singleton
     * @return the unsafe singleton
     * @throws Exception
     */
    public static Unsafe getUnsafe() throws Exception {
        if(INSTANCE == null) {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            Unsafe unsafe = (Unsafe) f.get(null);
            INSTANCE = unsafe;
            return unsafe;
        }

        return INSTANCE;

    }
}
