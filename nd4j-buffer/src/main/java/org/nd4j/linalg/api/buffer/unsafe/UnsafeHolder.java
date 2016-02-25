package org.nd4j.linalg.api.buffer.unsafe;

import sun.misc.Unsafe;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.ByteBuffer;

/**
 * Unsafe singleton holder
 * @author Adam Gibson
 */
public class UnsafeHolder {
    private static Unsafe INSTANCE;
    private static Field ADDRESS_FIELD;
    private static boolean is64Bit = System.getProperty("sun.arch.data.model").equals("64");
    private UnsafeHolder() {}


    /**
     * Returns true if the jvm is 64 bit
     * @return
     */
    public static boolean is64Bit() {
        return is64Bit;
    }

    /**
     * Returns the field singleton
     * used in the byte buffer
     * @return the address field in the bytebuffer
     * for reflection
     * @throws NoSuchFieldException
     */
    public static Field getAddressField() throws NoSuchFieldException {
        if(ADDRESS_FIELD == null) {
            ADDRESS_FIELD = getDeclaredFieldRecursive(ByteBuffer.class, "address");

        }
        return ADDRESS_FIELD;
    }

    private static Field getDeclaredFieldRecursive(final Class<?> root, final String fieldName) throws NoSuchFieldException {
        Class<?> type = root;

        do {
            try {
                return type.getDeclaredField(fieldName);
            } catch (NoSuchFieldException e) {
                type = type.getSuperclass();
            }
        } while ( type != null );

        throw new NoSuchFieldException(fieldName + " does not exist in " + root.getSimpleName() + " or any of its superclasses.");
    }

    /**
     * Unsafe singleton
     * @return the unsafe singleton
     * Reference:
     * https://github.com/LWJGL/lwjgl/blob/master/src/java/org/lwjgl/MemoryUtilSun.java#L73
     * @throws Exception
     */
    public static Unsafe getUnsafe() throws Exception {
        if(INSTANCE == null) {
            final Field[] fields = Unsafe.class.getDeclaredFields();

			/*
			Different runtimes use different names for the Unsafe singleton,
			so we cannot use .getDeclaredField and we scan instead. For example:
			Oracle: theUnsafe
			PERC : m_unsafe_instance
			Android: THE_ONE
			*/
            for ( Field field : fields ) {
                if ( !field.getType().equals(Unsafe.class) )
                    continue;

                final int modifiers = field.getModifiers();
                if ( !(Modifier.isStatic(modifiers) && Modifier.isFinal(modifiers)) )
                    continue;

                field.setAccessible(true);
                try {
                    INSTANCE = (Unsafe)field.get(null);
                } catch (IllegalAccessException e) {
                    // ignore
                }
                break;
            }

            throw new UnsupportedOperationException();

        }

        return INSTANCE;

    }
}
