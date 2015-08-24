package org.nd4j.bytebuddy.shape;

import net.bytebuddy.ByteBuddy;
import net.bytebuddy.dynamic.DynamicType;
import net.bytebuddy.dynamic.loading.ClassLoadingStrategy;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.matcher.ElementMatchers;
import org.junit.Test;


import static org.junit.Assert.*;

/**
 * @author Adam Gibson
 */
public class ShapeMapperTest  {
    @Test
    public void testShapeMapper() throws Exception {
        Implementation cImpl = ShapeMapper.getInd2Sub('c', 2);
        Implementation fImpl = ShapeMapper.getInd2Sub('f', 2);
        DynamicType.Unloaded<IndexMapper> c = new ByteBuddy()
                .subclass(IndexMapper.class).method(ElementMatchers.isDeclaredBy(IndexMapper.class))
                .intercept(cImpl)
                .make();
        DynamicType.Unloaded<IndexMapper> f = new ByteBuddy()
                .subclass(IndexMapper.class).method(ElementMatchers.isDeclaredBy(IndexMapper.class))
                .intercept(fImpl)
                .make();

        Class<?> dynamicType = c.load(IndexMapper.class.getClassLoader(), ClassLoadingStrategy.Default.WRAPPER)
                .getLoaded();
        Class<?> dynamicTypeF = f.load(IndexMapper.class.getClassLoader(), ClassLoadingStrategy.Default.WRAPPER)
                .getLoaded();

        IndexMapper testC = (IndexMapper) dynamicType.newInstance();
        IndexMapper testF = (IndexMapper) dynamicTypeF.newInstance();
        int n = 1000;
        long byteBuddyTotal = 0;
        for(int i = 0; i < n; i++) {
            long start = System.nanoTime();
            int[] cTest = testC.ind2sub(new int[]{2, 2}, 1, 4, 'c');
            long end = System.nanoTime();
            byteBuddyTotal += Math.abs((end - start));

        }

        byteBuddyTotal /= n;
        System.out.println("Took " + byteBuddyTotal);

        int[] cTest = testC.ind2sub(new int[]{2, 2}, 1, 4, 'c');
        int[] fTest = testF.ind2sub(new int[]{2, 2}, 1, 4, 'f');
        assertArrayEquals(new int[]{1,0},fTest);
        assertArrayEquals(new int[]{0,1},cTest);

    }



}
