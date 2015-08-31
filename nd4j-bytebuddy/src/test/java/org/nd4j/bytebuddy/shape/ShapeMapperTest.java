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


    @Test
    public void testOffsetMapper() throws Exception{
        OffsetMapper mapper = ShapeMapper.getOffsetMapperInstance(2);
        assertEquals(verifyImpl(0,new int[]{3,5},new int[]{4,1},new int[]{1,1}),mapper.getOffset(0,new int[]{3,5},new int[]{4,1},new int[]{1,1}));
        long oldImplTotal = 0;
        long newImplTotal = 0;
        int[] timingShape = {1,5,1,1};
        int[] timingStride = {4,1,1,1};
        int[] timingIndex = {1,1,1,1};
        for(int i = 0; i < 1000; i++) {
            long old = System.nanoTime();
            verifyImpl(0,timingShape,timingStride,timingIndex);
            long newTime = System.nanoTime();
            long delta = Math.abs(newTime - old);
            long oldDelta = delta;
            oldImplTotal += delta;
            old = System.nanoTime();
            mapper.getOffset(0,timingShape,timingStride,timingIndex);
            newTime = System.nanoTime();
            delta = Math.abs(newTime - old);
            newImplTotal += delta;
            System.out.println("Time for old was " + oldDelta + " while new was " + delta + " in nanoseconds at " + i);

        }

        oldImplTotal /= 1000;
        newImplTotal /= 1000;
        System.out.println("Time for old was " + oldImplTotal + " while new was " + newImplTotal + " in nanoseconds");
    }


    private int verifyImpl(int baseOffset,int[] shape,int[] stride,int[] indices) {
        int offset = 0;
        for(int i = 0; i < indices.length; i++) {
            /**
             * See:
             * http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
             * Basically if the size(i) is 1, the stride shouldn't be counted.
             */
            if(shape[i] == 1)
                continue;
            offset += indices[i] * stride[i];
        }


        return offset + baseOffset;

    }


}
