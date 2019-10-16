package org.nd4j.autodiff.internal;

import org.junit.Test;
import org.nd4j.autodiff.samediff.internal.DependencyList;
import org.nd4j.autodiff.samediff.internal.DependencyTracker;
import org.nd4j.autodiff.samediff.internal.IdentityDependencyTracker;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
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

        dt.addOrDependency("y", "x1", "x2");
        dl = dt.getDependencies("y");
        assertNull(dl.getDependencies());
        assertNotNull(dl.getOrDependencies());
        assertEquals(Collections.singletonList(new Pair<>("x1", "x2")), dl.getOrDependencies());
        dt.removeDependency("y", "x2");
        assertTrue(dt.isEmpty());
    }


    @Test
    public void testIdentityDependencyTracker(){
        IdentityDependencyTracker<INDArray, String> dt = new IdentityDependencyTracker<>();
        assertTrue(dt.isEmpty());

        INDArray y1 = Nd4j.scalar(0);
        INDArray y2 = Nd4j.scalar(0);
        String x1 = "x1";
        dt.addDependency(y1, x1);

        assertFalse(dt.hasNewAllSatisfied());
        assertTrue(dt.hasDependency(y1));
        assertFalse(dt.hasDependency(y2));
        assertFalse(dt.isSatisfied(x1));

        DependencyList<INDArray, String> dl = dt.getDependencies(y1);
        assertSame(y1, dl.getDependencyFor());      //Should be same object
        assertEquals(Collections.singletonList(x1), dl.getDependencies());
        assertNull(dl.getOrDependencies());


        //Mark as satisfied, check if it's added to list
        dt.markSatisfied(x1, true);
        assertTrue(dt.isSatisfied(x1));
        assertTrue(dt.hasNewAllSatisfied());
        INDArray get = dt.getNewAllSatisfied();
        assertSame(y1, get);
        assertFalse(dt.hasNewAllSatisfied());
    }

}
