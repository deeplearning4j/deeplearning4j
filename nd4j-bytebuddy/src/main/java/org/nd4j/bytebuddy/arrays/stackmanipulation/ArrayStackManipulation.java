package org.nd4j.bytebuddy.arrays.stackmanipulation;

import net.bytebuddy.implementation.bytecode.StackManipulation;
import net.bytebuddy.implementation.bytecode.collection.ArrayAccess;
import net.bytebuddy.pool.TypePool;

/**
 * Stack manipulations
 * for get for array manipulation (only for integers!)
 *
 * @author Adam Gibson
 */
public class ArrayStackManipulation {
    private static TypePool typePool = TypePool.Default.ofClassPath();
    private static StackManipulation store = ArrayAccess.of(typePool.describe("int").resolve()).store();
    private static StackManipulation load = ArrayAccess.of(typePool.describe("int").resolve()).load();

    public static StackManipulation store() {
        return store;
    }

    public static StackManipulation load() {
        return load;
    }

}
