package org.nd4j.linalg;

import junit.framework.TestCase;
import org.junit.After;
import org.junit.Before;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.UUID;

/**
 * Created by agibsoncccc on 5/11/15.
 */
public abstract class BaseNd4jTest extends TestCase {
    protected Nd4jBackend backend;

    public BaseNd4jTest(String name, Nd4jBackend backend) {
        super(name);
        this.backend = backend;
    }

    public BaseNd4jTest(Nd4jBackend backend) {
        this(backend.getClass().getName() + UUID.randomUUID().toString(),backend);

    }

    @Before
    public void before() {
        Nd4j nd4j = new Nd4j();
        nd4j.initWithBackend(backend);
    }

    @After
    public void after() {
        Nd4j nd4j = new Nd4j();
        nd4j.initWithBackend(backend);
    }

    @Override
    public String getName() {
        return getClass().getName();
    }


}
