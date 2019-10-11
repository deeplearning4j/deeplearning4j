package org.nd4j.autodiff.internal;

import org.junit.Test;
import org.nd4j.autodiff.samediff.internal.DependencyList;
import org.nd4j.autodiff.samediff.internal.DependencyTracker;
import org.nd4j.linalg.primitives.Pair;

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
        assertTrue(dt.isEmpty());


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
        assertTrue(dt.isEmpty());



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
        assertTrue(dt.hasZeroDependencyItem());
        assertEquals(Collections.singletonList("y"), dt.removeAllZeroDependencyItems());
        assertTrue(dt.isEmpty());



        //Dependee aliases (i.e., x -> y, with x1 == x2)
        assertFalse(dt.isDependeeAlias("x"));
        assertFalse(dt.isDependeeAlias("y"));
        dt.addDependeeAlias("y", "x");      //x is alias of y
        assertTrue(dt.isDependeeAlias("x"));
        assertFalse(dt.isDependeeAlias("y"));
        dt.addDependeeAlias("x", "z");      //z is alias of x; by extension, z is alias of y
        assertTrue(dt.isDependeeAlias("z"));
        assertTrue(dt.isDependeeAlias("x"));
        assertFalse(dt.isDependeeAlias("y"));
        assertEquals("y", dt.dependeeAliasGetUnderlying("x"));
        assertEquals("y", dt.dependeeAliasGetUnderlying("z"));
        dt.removeDependeeAlias("z");
        assertFalse(dt.isDependeeAlias("z"));
        dt.removeDependeeAlias("x");
        assertTrue(dt.isEmpty());

        //Dependent aliases  (i.e., x -> y, with y1 == y2)
        dt.addDependentAlias("y", "y2");      //y2 is alias of y
        assertTrue(dt.isDependentAlias("y2"));
        assertFalse(dt.isDependentAlias("y"));
        dt.addDependentAlias("y2", "y3");      //y3 is alias of y2; by extension, y3 is alias of y
        assertTrue(dt.isDependentAlias("y2"));
        assertTrue(dt.isDependentAlias("y3"));
        assertFalse(dt.isDependentAlias("y"));
        assertEquals("y", dt.dependentAliasGetUnderlying("y2"));
        assertEquals("y", dt.dependentAliasGetUnderlying("y3"));
        dt.removeDependentAlias("y3");
        assertTrue(dt.isDependentAlias("y2"));
        assertFalse(dt.isDependentAlias("y3"));
        dt.removeDependentAlias("y2");
        assertFalse(dt.isDependentAlias("y2"));
        assertTrue(dt.isEmpty());


        //Combination of dependent and dependee aliases
        dt.addDependeeAlias("x", "x2");
        dt.addDependentAlias("y", "y2");
        dt.addDependency("y2", "x2");           //x2 -> y2, but due to aliases equivalent to x -> y
        assertEquals("y", dt.dependentAliasGetUnderlying("y2"));
        assertEquals("x", dt.dependeeAliasGetUnderlying("x2"));
        dl = dt.getDependencies("y");
        assertEquals(Collections.singletonList("x"), dl.getDependencies());
        assertNull(dl.getOrDependencies());


        //Add dependency alias after a dependency is already defined
    }

}
