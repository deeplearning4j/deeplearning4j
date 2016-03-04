package org.deeplearning4j.ui;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author raver119@gmail.com
 */
public class UiConnectionInfoTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testGetFirstPart1() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder()
                .setPort(8080)
                .build();

        assertEquals("http://localhost:8080", info.getFirstPart());
    }

    @Test
    public void testGetFirstPart2() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder()
                .enableHttps(true)
                .setPort(8080)
                .build();

        assertEquals("https://localhost:8080", info.getFirstPart());
    }

    @Test
    public void testGetFirstPart3() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder()
                .setAddress("192.168.1.1")
                .enableHttps(true)
                .setPort(8082)
                .build();

        assertEquals("https://192.168.1.1:8082", info.getFirstPart());
    }


    @Test
    public void testGetSecondPart1() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder()
                .setAddress("192.168.1.1")
                .enableHttps(true)
                .setPort(8082)
                .setPath("www-data")
                .build();

        assertEquals("/www-data/", info.getSecondPart());
    }

    @Test
    public void testGetSecondPart2() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder()
                .setAddress("192.168.1.1")
                .enableHttps(true)
                .setPort(8082)
                .setPath("/www-data/tmp/")
                .build();

        assertEquals("/www-data/tmp/", info.getSecondPart());
    }

    @Test
    public void testGetSecondPart3() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder()
                .setAddress("192.168.1.1")
                .enableHttps(true)
                .setPort(8082)
                .setPath("/www-data/tmp")
                .build();

        assertEquals("/www-data/tmp/", info.getSecondPart());
    }

    @Test
    public void testGetSecondPart4() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder()
                .setAddress("192.168.1.1")
                .enableHttps(true)
                .setPort(8082)
                .setPath("/www-data//tmp")
                .build();

        assertEquals("/www-data/tmp/", info.getSecondPart());
    }

    @Test
    public void testGetSecondPart5() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder()
                .setAddress("192.168.1.1")
                .enableHttps(true)
                .setPort(8082)
                .setPath("/www-data//tmp")
                .build();

        assertEquals("/www-data/tmp/alpha/", info.getSecondPart("alpha"));
    }

    @Test
    public void testGetSecondPart6() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder()
                .setAddress("192.168.1.1")
                .enableHttps(true)
                .setPort(8082)
                .setPath("//www-data//tmp")
                .build();

        assertEquals("/www-data/tmp/alpha/", info.getSecondPart("/alpha/"));
    }

    @Test
    public void testGetSecondPart7() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder()
                .setAddress("192.168.1.1")
                .enableHttps(true)
                .setPort(8082)
                .setPath("//www-data//tmp")
                .build();

        assertEquals("/www-data/tmp/alpha/beta/", info.getSecondPart("/alpha//beta/"));
    }

    @Test
    public void testGetSecondPart8() throws Exception {
        UiConnectionInfo info = new UiConnectionInfo.Builder()
                .setAddress("192.168.1.1")
                .enableHttps(false)
                .setPort(8082)
                .setPath("/www-data//tmp")
                .build();

        assertEquals("http://192.168.1.1:8082/www-data/tmp/", info.getFullAddress());
    }
}