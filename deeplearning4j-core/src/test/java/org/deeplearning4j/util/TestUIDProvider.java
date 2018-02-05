package org.deeplearning4j.util;

import org.deeplearning4j.BaseDL4JTest;
import org.junit.Test;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

/**
 * Created by Alex on 26/06/2016.
 */
public class TestUIDProvider extends BaseDL4JTest {

    @Test
    public void testUIDProvider() {
        String jvmUID = UIDProvider.getJVMUID();
        String hardwareUID = UIDProvider.getHardwareUID();

        assertNotNull(jvmUID);
        assertNotNull(hardwareUID);

        assertTrue(!jvmUID.isEmpty());
        assertTrue(!hardwareUID.isEmpty());

        assertEquals(jvmUID, UIDProvider.getJVMUID());
        assertEquals(hardwareUID, UIDProvider.getHardwareUID());

        System.out.println("JVM uid:      " + jvmUID);
        System.out.println("Hardware uid: " + hardwareUID);
    }

}
