package org.deeplearning4j.models.glove.count;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by fartovii on 23.12.15.
 */
public class RoundCountTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testGet1() throws Exception {
        RoundCount count = new RoundCount(1);

        assertEquals(0, count.get());

        count.tick();
        assertEquals(1, count.get());

        count.tick();
        assertEquals(0, count.get());
    }

    @Test
    public void testGet2() throws Exception {
        RoundCount count = new RoundCount(3);

        assertEquals(0, count.get());

        count.tick();
        assertEquals(1, count.get());

        count.tick();
        assertEquals(2, count.get());

        count.tick();
        assertEquals(3, count.get());

        count.tick();
        assertEquals(0, count.get());
    }

    @Test
    public void testPrevious1() throws Exception {
        RoundCount count = new RoundCount(3);

        assertEquals(0, count.get());
        assertEquals(3, count.previous());

        count.tick();
        assertEquals(1, count.get());
        assertEquals(0, count.previous());

        count.tick();
        assertEquals(2, count.get());
        assertEquals(1, count.previous());

        count.tick();
        assertEquals(3, count.get());
        assertEquals(2, count.previous());

        count.tick();
        assertEquals(0, count.get());
        assertEquals(3, count.previous());
    }
}
