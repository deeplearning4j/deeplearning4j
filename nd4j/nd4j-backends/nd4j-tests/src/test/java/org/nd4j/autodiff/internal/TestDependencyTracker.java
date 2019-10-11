package org.nd4j.autodiff.internal;

import org.junit.Test;
import org.nd4j.autodiff.samediff.internal.DependencyList;
import org.nd4j.autodiff.samediff.internal.DependencyTracker;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;
import java.util.Collections;

import static junit.framework.TestCase.assertNotNull;
import static org.junit.Assert.*;

public class TestDependencyTracker {

    @Test
    public void testSimple(){

        DependencyTracker<String,String> dt = new DependencyTracker<>();

        dt.addDependency("y", "x");
        assertTrue(dt.hasDependency("y"));
        assertFalse(dt.hasDependency("x"));
        assertFalse(dt.hasDependency("z"));

        DependencyList<String,String> dl = dt.getDependencies("y");
        assertEquals("y", dl.getDependencyFor());
        assertNotNull(dl.getDependencies());
        assertNull(dl.getOrDependencies());
        assertEquals(Collections.singletonList("x"), dl.getDependencies());

        dt.removeDependency("y", "x");
        assertFalse(dt.hasDependency("y"));
        assertFalse(dt.hasDependency("x"));
        dl = dt.getDependencies("y");
        assertNull(dl.getDependencies());
        assertNull(dl.getOrDependencies());

        assertTrue(dt.hasZeroDependencyItem());
        assertEquals("y", dt.removeZeroDependencyItem());
        assertFalse(dt.hasZeroDependencyItem());


        //Or dep
        dt.addOrDependency("y", "x1", "x2");
        assertTrue(dt.hasDependency("y"));
        dl = dt.getDependencies("y");
        assertNull(dl.getDependencies());
        assertNotNull(dl.getOrDependencies());
        assertEquals(Collections.singletonList(new Pair<>("x1", "x2")), dl.getOrDependencies());

        dt.removeDependency("y", "x1");
        assertFalse(dt.hasDependency("y"));
        dl = dt.getDependencies("y");
        assertNull(dl.getDependencies());
        assertNull(dl.getOrDependencies());

        assertTrue(dt.hasZeroDependencyItem());
        assertEquals(Collections.singletonList("y"), dt.removeAllZeroDependencyItems());
        assertFalse(dt.hasZeroDependencyItem());

        dt.addOrDependency("y", "x1", "x2");
        dl = dt.getDependencies("y");
        assertNull(dl.getDependencies());
        assertNotNull(dl.getOrDependencies());
        assertEquals(Collections.singletonList(new Pair<>("x1", "x2")), dl.getOrDependencies());
        dt.removeDependency("y", "x2");

        assertTrue(dt.hasZeroDependencyItem());
        assertEquals("y", dt.removeZeroDependencyItem());
        assertFalse(dt.hasZeroDependencyItem());



        //Zero dep
        dt.addZeroDependencyItem("y");
        dt.addZeroDependencyItem("y");
        dt.addZeroDependencyItem("y");
        assertTrue(dt.hasZeroDependencyItem());
        assertEquals("y", dt.removeZeroDependencyItem());
        assertFalse(dt.hasZeroDependencyItem());

        dt.addZeroDependencyItem("y");
        dt.addZeroDependencyItem("y");
        dt.addZeroDependencyItem("y");
        assertEquals(Collections.singletonList("y"), dt.removeAllZeroDependencyItems());
        assertFalse(dt.hasZeroDependencyItem());


        dt.addZeroDependencyItem("y");
        dt.addDependency("y", "x");
        dt.removeDependency("y", "x");



        //Alias
        assertFalse(dt.isAlias("x"));
        assertFalse(dt.isAlias("y"));
        dt.addAlias("y", "x");      //x is alias of y
        assertTrue(dt.isAlias("x"));
        assertFalse(dt.isAlias("y"));
        dt.addAlias("x", "z");      //z is alias of x; by extension, z is alias of y
        assertTrue(dt.isAlias("z"));
        assertTrue(dt.isAlias("x"));
        assertFalse(dt.isAlias("y"));
        assertEquals("y", dt.aliasGetUnderlying("x"));
        assertEquals("y", dt.aliasGetUnderlying("z"));
        dt.removeAlias("z");
        assertFalse(dt.isAlias("z"));


    }

}
