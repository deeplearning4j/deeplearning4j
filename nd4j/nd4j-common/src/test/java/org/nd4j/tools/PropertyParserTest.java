package org.nd4j.tools;

import java.util.Properties;
import org.junit.After;
import org.junit.AfterClass;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Tests for PropertyParser
 *
 * @author gagatust
 */
public class PropertyParserTest {

    public PropertyParserTest() {
    }

    @BeforeClass
    public static void setUpClass() {
    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

    /**
     * Test of getProperties method, of class PropertyParser.
     */
    @Test
    public void testGetProperties() {

    }

    /**
     * Test of setProperties method, of class PropertyParser.
     */
    @Test
    public void testSetProperties() {

    }

    /**
     * Test of parseString method, of class PropertyParser.
     */
    @Test
    public void testParseString() {
        System.out.println("parseString");
        String expResult;
        String result;

        Properties props = new Properties();
        props.put("value1", "sTr1");
        props.put("value2", "str_2");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = "sTr1";
        result = instance.parseString("value1");
        assertEquals(expResult, result);

        expResult = "str_2";
        result = instance.parseString("value2");
        assertEquals(expResult, result);

        expResult = "";
        result = instance.parseString("empty");
        assertEquals(expResult, result);

        expResult = "abc";
        result = instance.parseString("str");
        assertEquals(expResult, result);

        expResult = "true";
        result = instance.parseString("boolean");
        assertEquals(expResult, result);

        expResult = "24.98";
        result = instance.parseString("float");
        assertEquals(expResult, result);

        expResult = "12";
        result = instance.parseString("int");
        assertEquals(expResult, result);

        expResult = "a";
        result = instance.parseString("char");
        assertEquals(expResult, result);

        try {
            instance.parseString("nonexistent");
            fail("no exception");
        } catch (NullPointerException e) {
        }
    }

    /**
     * Test of parseInt method, of class PropertyParser.
     */
    @Test
    public void testParseInt() {
        System.out.println("parseInt");
        int expResult;
        int result;

        Properties props = new Properties();
        props.put("value1", "432");
        props.put("value2", "-242");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = 432;
        result = instance.parseInt("value1");
        assertEquals(expResult, result);

        expResult = -242;
        result = instance.parseInt("value2");
        assertEquals(expResult, result);

        try {
            instance.parseInt("empty");
            fail("no exception");
        } catch (NumberFormatException e) {
        }

        try {
            instance.parseInt("str");
            fail("no exception");
        } catch (NumberFormatException e) {
        }

        try {
            instance.parseInt("boolean");
            assertEquals(expResult, result);
            fail("no exception");
        } catch (NumberFormatException e) {
        }

        try {
            instance.parseInt("float");
            fail("no exception");
        } catch (NumberFormatException e) {
        }

        expResult = 12;
        result = instance.parseInt("int");
        assertEquals(expResult, result);

        try {
            instance.parseInt("char");
            fail("no exception");
        } catch (NumberFormatException e) {
        }

        try {
            expResult = 0;
            result = instance.parseInt("nonexistent");
            fail("no exception");
            assertEquals(expResult, result);
        } catch (IllegalArgumentException e) {
        }
    }

    /**
     * Test of parseBoolean method, of class PropertyParser.
     */
    @Test
    public void testParseBoolean() {
        System.out.println("parseBoolean");
        boolean expResult;
        boolean result;

        Properties props = new Properties();
        props.put("value1", "true");
        props.put("value2", "false");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = true;
        result = instance.parseBoolean("value1");
        assertEquals(expResult, result);

        expResult = false;
        result = instance.parseBoolean("value2");
        assertEquals(expResult, result);

        expResult = false;
        result = instance.parseBoolean("empty");
        assertEquals(expResult, result);

        expResult = false;
        result = instance.parseBoolean("str");
        assertEquals(expResult, result);

        expResult = true;
        result = instance.parseBoolean("boolean");
        assertEquals(expResult, result);

        expResult = false;
        result = instance.parseBoolean("float");
        assertEquals(expResult, result);

        expResult = false;
        result = instance.parseBoolean("int");
        assertEquals(expResult, result);

        expResult = false;
        result = instance.parseBoolean("char");
        assertEquals(expResult, result);

        try {
            expResult = false;
            result = instance.parseBoolean("nonexistent");
            fail("no exception");
            assertEquals(expResult, result);
        } catch (IllegalArgumentException e) {
        }
    }

    /**
     * Test of parseDouble method, of class PropertyParser.
     */
    @Test
    public void testParseFloat() {
        System.out.println("parseFloat");
        double expResult;
        double result;

        Properties props = new Properties();
        props.put("value1", "12345.6789");
        props.put("value2", "-9000.001");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = 12345.6789f;
        result = instance.parseFloat("value1");
        assertEquals(expResult, result, 0);

        expResult = -9000.001f;
        result = instance.parseFloat("value2");
        assertEquals(expResult, result, 0);

        try {
            instance.parseFloat("empty");
            fail("no exception");
        } catch (IllegalArgumentException e) {
        }

        try {
            instance.parseFloat("str");
            fail("no exception");
        } catch (IllegalArgumentException e) {
        }

        try {
            instance.parseFloat("boolean");
            fail("no exception");
        } catch (IllegalArgumentException e) {
        }

        expResult = 24.98f;
        result = instance.parseFloat("float");
        assertEquals(expResult, result, 0);

        expResult = 12f;
        result = instance.parseFloat("int");
        assertEquals(expResult, result, 0);

        try {
            instance.parseFloat("char");
            fail("no exception");
        } catch (IllegalArgumentException e) {
        }

        try {
            instance.parseFloat("nonexistent");
            fail("no exception");
        } catch (NullPointerException e) {
        }
    }

    /**
     * Test of parseDouble method, of class PropertyParser.
     */
    @Test
    public void testParseDouble() {
        System.out.println("parseDouble");
        double expResult;
        double result;

        Properties props = new Properties();
        props.put("value1", "12345.6789");
        props.put("value2", "-9000.001");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = 12345.6789;
        result = instance.parseDouble("value1");
        assertEquals(expResult, result, 0);

        expResult = -9000.001;
        result = instance.parseDouble("value2");
        assertEquals(expResult, result, 0);

        try {
            instance.parseDouble("empty");
            fail("no exception");
        } catch (IllegalArgumentException e) {
        }

        try {
            instance.parseDouble("str");
            fail("no exception");
        } catch (IllegalArgumentException e) {
        }

        try {
            instance.parseDouble("boolean");
            fail("no exception");
        } catch (IllegalArgumentException e) {
        }

        expResult = 24.98;
        result = instance.parseDouble("float");
        assertEquals(expResult, result, 0);

        expResult = 12;
        result = instance.parseDouble("int");
        assertEquals(expResult, result, 0);

        try {
            instance.parseDouble("char");
            fail("no exception");
        } catch (IllegalArgumentException e) {
        }

        try {
            instance.parseDouble("nonexistent");
            fail("no exception");
        } catch (NullPointerException e) {
        }
    }

    /**
     * Test of parseLong method, of class PropertyParser.
     */
    @Test
    public void testParseLong() {
        System.out.println("parseLong");
        long expResult;
        long result;

        Properties props = new Properties();
        props.put("value1", "12345678900");
        props.put("value2", "-9000001");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = 12345678900L;
        result = instance.parseLong("value1");
        assertEquals(expResult, result);

        expResult = -9000001L;
        result = instance.parseLong("value2");
        assertEquals(expResult, result);

        try {
            instance.parseLong("empty");
            fail("no exception");
        } catch (IllegalArgumentException e) {
        }

        try {
            instance.parseLong("str");
            fail("no exception");
        } catch (IllegalArgumentException e) {
        }

        try {
            instance.parseLong("boolean");
            fail("no exception");
        } catch (IllegalArgumentException e) {
        }

        try {
            instance.parseLong("float");
            fail("no exception");
        } catch (IllegalArgumentException e) {
        }

        expResult = 12L;
        result = instance.parseLong("int");
        assertEquals(expResult, result);

        try {
            instance.parseLong("char");
            fail("no exception");
        } catch (IllegalArgumentException e) {
        }

        try {
            instance.parseLong("nonexistent");
            fail("no exception");
        } catch (IllegalArgumentException e) {
        }
    }

    /**
     * Test of parseChar method, of class PropertyParser.
     */
    @Test
    public void testParseChar() {
        System.out.println("parseChar");
        char expResult;
        char result;

        Properties props = new Properties();
        props.put("value1", "b");
        props.put("value2", "c");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = 'b';
        result = instance.parseChar("value1");
        assertEquals(expResult, result);

        expResult = 'c';
        result = instance.parseChar("value2");
        assertEquals(expResult, result);

        try {
            instance.parseChar("empty");
        } catch (IllegalArgumentException e) {
        }

        try {
            instance.parseChar("str");
        } catch (IllegalArgumentException e) {
        }

        try {
            instance.parseChar("boolean");
        } catch (IllegalArgumentException e) {
        }

        try {
            instance.parseChar("float");
        } catch (IllegalArgumentException e) {
        }

        try {
            instance.parseChar("int");
        } catch (IllegalArgumentException e) {
        }

        expResult = 'a';
        result = instance.parseChar("char");
        assertEquals(expResult, result);

        try {
            instance.parseChar("nonexistent");
            fail("no exception");
            assertEquals(expResult, result);
        } catch (NullPointerException e) {
        }
    }

    /**
     * Test of toString method, of class PropertyParser.
     */
    @Test
    public void testToString_String() {
        System.out.println("toString");
        String expResult;
        String result;

        Properties props = new Properties();
        props.put("value1", "sTr1");
        props.put("value2", "str_2");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = "sTr1";
        result = instance.toString("value1");
        assertEquals(expResult, result);

        expResult = "str_2";
        result = instance.toString("value2");
        assertEquals(expResult, result);

        expResult = "";
        result = instance.toString("empty");
        assertEquals(expResult, result);

        expResult = "abc";
        result = instance.toString("str");
        assertEquals(expResult, result);

        expResult = "true";
        result = instance.toString("boolean");
        assertEquals(expResult, result);

        expResult = "24.98";
        result = instance.toString("float");
        assertEquals(expResult, result);

        expResult = "12";
        result = instance.toString("int");
        assertEquals(expResult, result);

        expResult = "a";
        result = instance.toString("char");
        assertEquals(expResult, result);

        expResult = "";
        result = instance.toString("nonexistent");
        assertEquals(expResult, result);
    }

    /**
     * Test of toInt method, of class PropertyParser.
     */
    @Test
    public void testToInt_String() {
        System.out.println("toInt");
        int expResult;
        int result;

        Properties props = new Properties();
        props.put("value1", "123");
        props.put("value2", "-54");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = 123;
        result = instance.toInt("value1");
        assertEquals(expResult, result);

        expResult = -54;
        result = instance.toInt("value2");
        assertEquals(expResult, result);

        expResult = 0;
        result = instance.toInt("empty");
        assertEquals(expResult, result);

        expResult = 0;
        result = instance.toInt("str");
        assertEquals(expResult, result);

        expResult = 0;
        result = instance.toInt("boolean");
        assertEquals(expResult, result);

        expResult = 0;
        result = instance.toInt("float");
        assertEquals(expResult, result);

        expResult = 12;
        result = instance.toInt("int");
        assertEquals(expResult, result);

        expResult = 0;
        result = instance.toInt("char");
        assertEquals(expResult, result);

        expResult = 0;
        result = instance.toInt("nonexistent");
        assertEquals(expResult, result);
    }

    /**
     * Test of toBoolean method, of class PropertyParser.
     */
    @Test
    public void testToBoolean_String() {
        System.out.println("toBoolean");
        boolean expResult;
        boolean result;

        Properties props = new Properties();
        props.put("value1", "true");
        props.put("value2", "false");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = true;
        result = instance.toBoolean("value1");
        assertEquals(expResult, result);

        expResult = false;
        result = instance.toBoolean("value2");
        assertEquals(expResult, result);

        expResult = false;
        result = instance.toBoolean("empty");
        assertEquals(expResult, result);

        expResult = false;
        result = instance.toBoolean("str");
        assertEquals(expResult, result);

        expResult = true;
        result = instance.toBoolean("boolean");
        assertEquals(expResult, result);

        expResult = false;
        result = instance.toBoolean("float");
        assertEquals(expResult, result);

        expResult = false;
        result = instance.toBoolean("int");
        assertEquals(expResult, result);

        expResult = false;
        result = instance.toBoolean("char");
        assertEquals(expResult, result);

        expResult = false;
        result = instance.toBoolean("nonexistent");
        assertEquals(expResult, result);
    }

    /**
     * Test of toDouble method, of class PropertyParser.
     */
    @Test
    public void testToFloat_String() {
        System.out.println("toFloat");
        float expResult;
        float result;

        Properties props = new Properties();
        props.put("value1", "12345.6789");
        props.put("value2", "-9000.001");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = 12345.6789f;
        result = instance.toFloat("value1");
        assertEquals(expResult, result, 0f);

        expResult = -9000.001f;
        result = instance.toFloat("value2");
        assertEquals(expResult, result, 0f);

        expResult = 0f;
        result = instance.toFloat("empty");
        assertEquals(expResult, result, 0f);

        expResult = 0f;
        result = instance.toFloat("str");
        assertEquals(expResult, result, 0f);

        expResult = 0f;
        result = instance.toFloat("boolean");
        assertEquals(expResult, result, 0f);

        expResult = 24.98f;
        result = instance.toFloat("float");
        assertEquals(expResult, result, 0f);

        expResult = 12f;
        result = instance.toFloat("int");
        assertEquals(expResult, result, 0f);

        expResult = 0f;
        result = instance.toFloat("char");
        assertEquals(expResult, result, 0f);

        expResult = 0f;
        result = instance.toFloat("nonexistent");
        assertEquals(expResult, result, 0f);
    }

    /**
     * Test of toDouble method, of class PropertyParser.
     */
    @Test
    public void testToDouble_String() {
        System.out.println("toDouble");
        double expResult;
        double result;

        Properties props = new Properties();
        props.put("value1", "12345.6789");
        props.put("value2", "-9000.001");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = 12345.6789;
        result = instance.toDouble("value1");
        assertEquals(expResult, result, 0);

        expResult = -9000.001;
        result = instance.toDouble("value2");
        assertEquals(expResult, result, 0);

        expResult = 0;
        result = instance.toDouble("empty");
        assertEquals(expResult, result, 0);

        expResult = 0;
        result = instance.toDouble("str");
        assertEquals(expResult, result, 0);

        expResult = 0;
        result = instance.toDouble("boolean");
        assertEquals(expResult, result, 0);

        expResult = 24.98;
        result = instance.toDouble("float");
        assertEquals(expResult, result, 0);

        expResult = 12;
        result = instance.toDouble("int");
        assertEquals(expResult, result, 0);

        expResult = 0;
        result = instance.toDouble("char");
        assertEquals(expResult, result, 0);

        expResult = 0;
        result = instance.toDouble("nonexistent");
        assertEquals(expResult, result, 0);
    }

    /**
     * Test of toLong method, of class PropertyParser.
     */
    @Test
    public void testToLong_String() {
        System.out.println("toLong");
        long expResult;
        long result;

        Properties props = new Properties();
        props.put("value1", "12345678900");
        props.put("value2", "-9000001");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = 12345678900L;
        result = instance.toLong("value1");
        assertEquals(expResult, result);

        expResult = -9000001L;
        result = instance.toLong("value2");
        assertEquals(expResult, result);

        expResult = 0L;
        result = instance.toLong("empty");
        assertEquals(expResult, result);

        expResult = 0L;
        result = instance.toLong("str");
        assertEquals(expResult, result);

        expResult = 0L;
        result = instance.toLong("boolean");
        assertEquals(expResult, result);

        expResult = 0L;
        result = instance.toLong("float");
        assertEquals(expResult, result);

        expResult = 12L;
        result = instance.toLong("int");
        assertEquals(expResult, result);

        expResult = 0L;
        result = instance.toLong("char");
        assertEquals(expResult, result);

        expResult = 0L;
        result = instance.toLong("nonexistent");
        assertEquals(expResult, result);
    }

    /**
     * Test of toChar method, of class PropertyParser.
     */
    @Test
    public void testToChar_String() {
        System.out.println("toChar");
        char expResult;
        char result;

        Properties props = new Properties();
        props.put("value1", "f");
        props.put("value2", "w");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = 'f';
        result = instance.toChar("value1");
        assertEquals(expResult, result);

        expResult = 'w';
        result = instance.toChar("value2");
        assertEquals(expResult, result);

        expResult = '\u0000';
        result = instance.toChar("empty");
        assertEquals(expResult, result);

        expResult = '\u0000';
        result = instance.toChar("str");
        assertEquals(expResult, result);

        expResult = '\u0000';
        result = instance.toChar("boolean");
        assertEquals(expResult, result);

        expResult = '\u0000';
        result = instance.toChar("float");
        assertEquals(expResult, result);

        expResult = '\u0000';
        result = instance.toChar("int");
        assertEquals(expResult, result);

        expResult = 'a';
        result = instance.toChar("char");
        assertEquals(expResult, result);

        expResult = '\u0000';
        result = instance.toChar("nonexistent");
        assertEquals(expResult, result);
    }

    /**
     * Test of toString method, of class PropertyParser.
     */
    @Test
    public void testToString_String_String() {
        System.out.println("toString");
        String expResult;
        String result;

        Properties props = new Properties();
        props.put("value1", "sTr1");
        props.put("value2", "str_2");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = "sTr1";
        result = instance.toString("value1", "defStr");
        assertEquals(expResult, result);

        expResult = "str_2";
        result = instance.toString("value2", "defStr");
        assertEquals(expResult, result);

        expResult = "";
        result = instance.toString("empty", "defStr");
        assertEquals(expResult, result);

        expResult = "abc";
        result = instance.toString("str", "defStr");
        assertEquals(expResult, result);

        expResult = "true";
        result = instance.toString("boolean", "defStr");
        assertEquals(expResult, result);

        expResult = "24.98";
        result = instance.toString("float", "defStr");
        assertEquals(expResult, result);

        expResult = "12";
        result = instance.toString("int", "defStr");
        assertEquals(expResult, result);

        expResult = "a";
        result = instance.toString("char", "defStr");
        assertEquals(expResult, result);

        expResult = "defStr";
        result = instance.toString("nonexistent", "defStr");
        assertEquals(expResult, result);
    }

    /**
     * Test of toInt method, of class PropertyParser.
     */
    @Test
    public void testToInt_String_int() {
        System.out.println("toInt");
        int expResult;
        int result;

        Properties props = new Properties();
        props.put("value1", "123");
        props.put("value2", "-54");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = 123;
        result = instance.toInt("value1", 17);
        assertEquals(expResult, result);

        expResult = -54;
        result = instance.toInt("value2", 17);
        assertEquals(expResult, result);

        expResult = 17;
        result = instance.toInt("empty", 17);
        assertEquals(expResult, result);

        expResult = 17;
        result = instance.toInt("str", 17);
        assertEquals(expResult, result);

        expResult = 17;
        result = instance.toInt("boolean", 17);
        assertEquals(expResult, result);

        expResult = 17;
        result = instance.toInt("float", 17);
        assertEquals(expResult, result);

        expResult = 12;
        result = instance.toInt("int", 17);
        assertEquals(expResult, result);

        expResult = 17;
        result = instance.toInt("char", 17);
        assertEquals(expResult, result);

        expResult = 17;
        result = instance.toInt("nonexistent", 17);
        assertEquals(expResult, result);
    }

    /**
     * Test of toBoolean method, of class PropertyParser.
     */
    @Test
    public void testToBoolean_String_boolean() {
        System.out.println("toBoolean");

        boolean expResult;
        boolean result;

        Properties props = new Properties();
        props.put("value1", "true");
        props.put("value2", "false");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = true;
        result = instance.toBoolean("value1", true);
        assertEquals(expResult, result);

        expResult = false;
        result = instance.toBoolean("value2", true);
        assertEquals(expResult, result);

        expResult = false;
        result = instance.toBoolean("empty", true);
        assertEquals(expResult, result);

        expResult = false;
        result = instance.toBoolean("str", true);
        assertEquals(expResult, result);

        expResult = true;
        result = instance.toBoolean("boolean", true);
        assertEquals(expResult, result);

        expResult = false;
        result = instance.toBoolean("float", true);
        assertEquals(expResult, result);

        expResult = false;
        result = instance.toBoolean("int", true);
        assertEquals(expResult, result);

        expResult = false;
        result = instance.toBoolean("char", true);
        assertEquals(expResult, result);

        expResult = true;
        result = instance.toBoolean("nonexistent", true);
        assertEquals(expResult, result);
    }

    /**
     * Test of toDouble method, of class PropertyParser.
     */
    @Test
    public void testToFloat_String_float() {
        System.out.println("toFloat");
        float expResult;
        float result;

        Properties props = new Properties();
        props.put("value1", "12345.6789");
        props.put("value2", "-9000.001");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = 12345.6789f;
        result = instance.toFloat("value1", 0.123f);
        assertEquals(expResult, result, 0);

        expResult = -9000.001f;
        result = instance.toFloat("value2", 0.123f);
        assertEquals(expResult, result, 0);

        expResult = 0.123f;
        result = instance.toFloat("empty", 0.123f);
        assertEquals(expResult, result, 0);

        expResult = 0.123f;
        result = instance.toFloat("str", 0.123f);
        assertEquals(expResult, result, 0);

        expResult = 0.123f;
        result = instance.toFloat("boolean", 0.123f);
        assertEquals(expResult, result, 0);

        expResult = 24.98f;
        result = instance.toFloat("float", 0.123f);
        assertEquals(expResult, result, 0);

        expResult = 12;
        result = instance.toFloat("int", 0.123f);
        assertEquals(expResult, result, 0);

        expResult = 0.123f;
        result = instance.toFloat("char", 0.123f);
        assertEquals(expResult, result, 0);

        expResult = 0.123f;
        result = instance.toFloat("nonexistent", 0.123f);
        assertEquals(expResult, result, 0);
    }

    /**
     * Test of toDouble method, of class PropertyParser.
     */
    @Test
    public void testToDouble_String_double() {
        System.out.println("toDouble");
        double expResult;
        double result;

        Properties props = new Properties();
        props.put("value1", "12345.6789");
        props.put("value2", "-9000.001");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = 12345.6789;
        result = instance.toDouble("value1", 0.123);
        assertEquals(expResult, result, 0);

        expResult = -9000.001;
        result = instance.toDouble("value2", 0.123);
        assertEquals(expResult, result, 0);

        expResult = 0.123;
        result = instance.toDouble("empty", 0.123);
        assertEquals(expResult, result, 0);

        expResult = 0.123;
        result = instance.toDouble("str", 0.123);
        assertEquals(expResult, result, 0);

        expResult = 0.123;
        result = instance.toDouble("boolean", 0.123);
        assertEquals(expResult, result, 0);

        expResult = 24.98;
        result = instance.toDouble("float", 0.123);
        assertEquals(expResult, result, 0);

        expResult = 12;
        result = instance.toDouble("int", 0.123);
        assertEquals(expResult, result, 0);

        expResult = 0.123;
        result = instance.toDouble("char", 0.123);
        assertEquals(expResult, result, 0);

        expResult = 0.123;
        result = instance.toDouble("nonexistent", 0.123);
        assertEquals(expResult, result, 0);
    }

    /**
     * Test of toLong method, of class PropertyParser.
     */
    @Test
    public void testToLong_String_long() {
        System.out.println("toLong");
        long expResult;
        long result;

        Properties props = new Properties();
        props.put("value1", "12345678900");
        props.put("value2", "-9000001");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = 12345678900L;
        result = instance.toLong("value1", 3L);
        assertEquals(expResult, result);

        expResult = -9000001L;
        result = instance.toLong("value2", 3L);
        assertEquals(expResult, result);

        expResult = 3L;
        result = instance.toLong("empty", 3L);
        assertEquals(expResult, result);

        expResult = 3L;
        result = instance.toLong("str", 3L);
        assertEquals(expResult, result);

        expResult = 3L;
        result = instance.toLong("boolean", 3L);
        assertEquals(expResult, result);

        expResult = 3L;
        result = instance.toLong("float", 3L);
        assertEquals(expResult, result);

        expResult = 12L;
        result = instance.toLong("int", 3L);
        assertEquals(expResult, result);

        expResult = 3L;
        result = instance.toLong("char", 3L);
        assertEquals(expResult, result);

        expResult = 3L;
        result = instance.toLong("nonexistent", 3L);
        assertEquals(expResult, result);
    }

    /**
     * Test of toChar method, of class PropertyParser.
     */
    @Test
    public void testToChar_String_char() {
        System.out.println("toChar");
        char expResult;
        char result;

        Properties props = new Properties();
        props.put("value1", "f");
        props.put("value2", "w");
        props.put("empty", "");
        props.put("str", "abc");
        props.put("boolean", "true");
        props.put("float", "24.98");
        props.put("int", "12");
        props.put("char", "a");
        PropertyParser instance = new PropertyParser(props);

        expResult = 'f';
        result = instance.toChar("value1", 't');
        assertEquals(expResult, result);

        expResult = 'w';
        result = instance.toChar("value2", 't');
        assertEquals(expResult, result);

        expResult = 't';
        result = instance.toChar("empty", 't');
        assertEquals(expResult, result);

        expResult = 't';
        result = instance.toChar("str", 't');
        assertEquals(expResult, result);

        expResult = 't';
        result = instance.toChar("boolean", 't');
        assertEquals(expResult, result);

        expResult = 't';
        result = instance.toChar("float", 't');
        assertEquals(expResult, result);

        expResult = 't';
        result = instance.toChar("int", 't');
        assertEquals(expResult, result);

        expResult = 'a';
        result = instance.toChar("char", 't');
        assertEquals(expResult, result);

        expResult = 't';
        result = instance.toChar("nonexistent", 't');
        assertEquals(expResult, result);
    }
    
}
