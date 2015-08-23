package org.nd4j.bytebuddy.shape;

import net.bytebuddy.ByteBuddy;
import net.bytebuddy.dynamic.DynamicType;
import net.bytebuddy.dynamic.loading.ClassLoadingStrategy;
import net.bytebuddy.implementation.Implementation;
import net.bytebuddy.matcher.ElementMatchers;
import org.junit.Test;

import java.io.File;

/**
 * @author Adam Gibson
 */
public class ShapeMapperTest  {
    @Test
    public void testShapeMapper() throws Exception {
        Implementation impl = ShapeMapper.getInd2Sub('c', 4);
        DynamicType.Unloaded<IndexMapper> arr = new ByteBuddy()
                .subclass(IndexMapper.class).method(ElementMatchers.isDeclaredBy(IndexMapper.class))
                .intercept(impl)
                .make();

        arr.saveIn(new File("/home/agibsonccc/code/nd4j/indexmapper"));

        Class<?> dynamicType = arr.load(IndexMapper.class.getClassLoader(), ClassLoadingStrategy.Default.WRAPPER)
                .getLoaded();

        IndexMapper test = (IndexMapper) dynamicType.newInstance();

    }

}
