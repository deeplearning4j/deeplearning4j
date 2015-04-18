package org.nd4j.linalg.factory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;
import org.nd4j.linalg.factory.Nd4jBackend.NoAvailableBackendException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;
import static org.junit.Assert.*;

public class Nd4jBackendTest {
    private static Logger log = LoggerFactory.getLogger(Nd4jBackendTest.class);

    @Test
    public void testPrioritization() {
        BackendBehavior behavior = new BackendBehavior();
        behavior.availabilityMap.put(Backend2.class, false);
        behavior.availabilityMap.put(Backend1.class, false);
        behaviorHolder.set(behavior);

        try {
            Nd4jBackend.load();
            fail();
        } catch (NoAvailableBackendException e) {
        }
        assertArrayEquals(new Object[] { Backend2.class, Backend1.class },
                behavior.invocationList.toArray());
    }

    @Test()
    public void testAvailability() {
        BackendBehavior behavior = new BackendBehavior();
        behavior.availabilityMap.put(Backend2.class, false);
        behavior.availabilityMap.put(Backend1.class, true);
        behaviorHolder.set(behavior);

        try {
            Nd4jBackend backend = Nd4jBackend.load();
            assertNotNull(backend);
            assertEquals(Backend1.class, backend.getClass());
        } catch (NoAvailableBackendException e) {
            fail();
        }
    }

    @Test(expected = Nd4jBackend.NoAvailableBackendException.class)
    public void testNoAvailableBackend() throws NoAvailableBackendException {
        BackendBehavior behavior = new BackendBehavior();
        behavior.availabilityMap.put(Backend2.class, false);
        behavior.availabilityMap.put(Backend1.class, false);
        behaviorHolder.set(behavior);

        Nd4jBackend.load();
    }

    private static final ThreadLocal<BackendBehavior> behaviorHolder = new ThreadLocal<>();

    private static class BackendBehavior {
        Map<Class<? extends Nd4jBackend>, Boolean> availabilityMap = new HashMap<>();
        List<Class<? extends Nd4jBackend>> invocationList = new ArrayList<>();
    }

    public static abstract class TestBackend extends Nd4jBackend {

        private int priority;

        protected TestBackend(int priority) {
            this.priority = priority;
        }

        @Override
        public int getPriority() {
            return this.priority;
        }

        @Override
        public boolean isAvailable() {
            BackendBehavior behavior = behaviorHolder.get();
            assert (behavior != null);
            assert (behavior.availabilityMap.containsKey(this.getClass()));

            behavior.invocationList.add(this.getClass());
            return behavior.availabilityMap.get(this.getClass());
        }

        @Override
        public Resource getConfigurationResource() {
            throw new UnsupportedOperationException();
        }

    }

    public static class Backend1 extends TestBackend {
        public Backend1() {
            super(1);
        }
    }

    public static class Backend2 extends TestBackend {
        public Backend2() {
            super(2);
        }
    }

}
