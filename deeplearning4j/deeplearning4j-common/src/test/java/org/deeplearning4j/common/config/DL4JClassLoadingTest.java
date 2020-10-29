package org.deeplearning4j.common.config;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import org.deeplearning4j.common.config.dummies.TestAbstract;
import org.junit.Test;

public class DL4JClassLoadingTest {
    private static final String PACKAGE_PREFIX = "org.deeplearning4j.common.config.dummies.";

    @Test
    public void testCreateNewInstance_constructorWithoutArguments() {

        /* Given */
        String className = PACKAGE_PREFIX + "TestDummy";

        /* When */
        Object instance = DL4JClassLoading.createNewInstance(className);

        /* Then */
        assertNotNull(instance);
        assertEquals(className, instance.getClass().getName());
    }

    @Test
    public void testCreateNewInstance_constructorWithArgument_implicitArgumentTypes() {

        /* Given */
        String className = PACKAGE_PREFIX + "TestColor";

        /* When */
        TestAbstract instance = DL4JClassLoading.createNewInstance(className, TestAbstract.class, "white");

        /* Then */
        assertNotNull(instance);
        assertEquals(className, instance.getClass().getName());
    }

    @Test
    public void testCreateNewInstance_constructorWithArgument_explicitArgumentTypes() {

        /* Given */
        String colorClassName = PACKAGE_PREFIX + "TestColor";
        String rectangleClassName = PACKAGE_PREFIX + "TestRectangle";

        /* When */
        TestAbstract color = DL4JClassLoading.createNewInstance(
                colorClassName,
                Object.class,
                new Class<?>[]{ int.class, int.class, int.class },
                45, 175, 200);

        TestAbstract rectangle = DL4JClassLoading.createNewInstance(
                rectangleClassName,
                Object.class,
                new Class<?>[]{ int.class, int.class, TestAbstract.class },
                10, 15, color);

        /* Then */
        assertNotNull(color);
        assertEquals(colorClassName, color.getClass().getName());

        assertNotNull(rectangle);
        assertEquals(rectangleClassName, rectangle.getClass().getName());
    }
}
