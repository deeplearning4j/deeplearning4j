package org.nd4j.tools;

import java.util.Properties;

/**
 * PropertyParser
 *
 * @author gagatust
 */
public class PropertyParser {

    private Properties properties;

    public PropertyParser(Properties properties) {
        this.properties = properties;
    }

    public Properties getProperties() {
        return properties;
    }

    public void setProperties(Properties properties) {
        this.properties = properties;
    }

    /**
     * Parse property.
     *
     * @param name property name
     * @return property
     */
    public String parseString(String name) {
        String property = getProperties().getProperty(name);
        if (property == null) {
            throw new NullPointerException();
        }
        return property;
    }

    /**
     * Parse property.
     *
     * @param name property name
     * @return property
     */
    public int parseInt(String name) {
        return Integer.parseInt(getProperties().getProperty(name));
    }

    /**
     * Parse property.
     *
     * @param name property name
     * @return property
     */
    public boolean parseBoolean(String name) {
        String property = getProperties().getProperty(name);
        if (property == null) {
            throw new IllegalArgumentException();
        }
        return Boolean.parseBoolean(property);
    }

    /**
     * Parse property.
     *
     * @param name property name
     * @return property
     */
    public float parseFloat(String name) {
        return Float.parseFloat(getProperties().getProperty(name));
    }

    /**
     * Parse property.
     *
     * @param name property name
     * @return property
     */
    public double parseDouble(String name) {
        return Double.parseDouble(getProperties().getProperty(name));
    }

    /**
     * Parse property.
     *
     * @param name property name
     * @return property
     */
    public long parseLong(String name) {
        return Long.parseLong(getProperties().getProperty(name));
    }

    /**
     * Parse property.
     *
     * @param name property name
     * @return property
     */
    public char parseChar(String name) {
        String property = getProperties().getProperty(name);
        if (property.length() != 1) {
            throw new IllegalArgumentException(name + " property is't char");
        }
        return property.charAt(0);
    }

    /**
     * Get property. The method returns the default value if the property is not parsed.
     *
     * @param name property name
     * @return property
     */
    public String toString(String name) {
        return toString(name, "");
    }

    /**
     * Get property. The method returns the default value if the property is not parsed.
     *
     * @param name property name
     * @return property
     */
    public int toInt(String name) {
        return toInt(name, 0);
    }

    /**
     * Get property. The method returns the default value if the property is not parsed.
     *
     * @param name property name
     * @return property
     */
    public boolean toBoolean(String name) {
        return toBoolean(name, false);
    }

    /**
     * Get property. The method returns the default value if the property is not parsed.
     *
     * @param name property name
     * @return property
     */
    public float toFloat(String name) {
        return toFloat(name, 0.0f);
    }

    /**
     * Get property. The method returns the default value if the property is not parsed.
     *
     * @param name property name
     * @return property
     */
    public double toDouble(String name) {
        return toDouble(name, 0.0);
    }

    /**
     * Get property. The method returns the default value if the property is not parsed.
     *
     * @param name property name
     * @return property
     */
    public long toLong(String name) {
        return toLong(name, 0);
    }

    /**
     * Get property. The method returns the default value if the property is not parsed.
     *
     * @param name property name
     * @return property
     */
    public char toChar(String name) {
        return toChar(name, '\u0000');
    }

    /**
     * Get property. The method returns the default value if the property is not parsed.
     *
     * @param name property name
     * @param defaultValue default value
     * @return property
     */
    public String toString(String name, String defaultValue) {
        try {
            return parseString(name);
        } catch (Exception e) {
            return defaultValue;
        }
    }

    /**
     * Get property. The method returns the default value if the property is not parsed.
     *
     * @param name property name
     * @param defaultValue default value
     * @return property
     */
    public int toInt(String name, int defaultValue) {
        try {
            return parseInt(name);
        } catch (Exception e) {
            return defaultValue;
        }
    }

    /**
     * Get property. The method returns the default value if the property is not parsed.
     *
     * @param name property name
     * @param defaultValue default value
     * @return property
     */
    public boolean toBoolean(String name, boolean defaultValue) {
        try {
            return parseBoolean(name);
        } catch (Exception e) {
            return defaultValue;
        }
    }

    /**
     * Get property. The method returns the default value if the property is not parsed.
     *
     * @param name property name
     * @param defaultValue default value
     * @return property
     */
    public float toFloat(String name, float defaultValue) {
        try {
            return parseFloat(name);
        } catch (Exception e) {
            return defaultValue;
        }
    }

    /**
     * Get property. The method returns the default value if the property is not parsed.
     *
     * @param name property name
     * @param defaultValue default value
     * @return property
     */
    public double toDouble(String name, double defaultValue) {
        try {
            return parseDouble(name);
        } catch (Exception e) {
            return defaultValue;
        }
    }

    /**
     * Get property. The method returns the default value if the property is not parsed.
     *
     * @param name property name
     * @param defaultValue default value
     * @return property
     */
    public long toLong(String name, long defaultValue) {
        try {
            return parseLong(name);
        } catch (Exception e) {
            return defaultValue;
        }
    }

    /**
     * Get property. The method returns the default value if the property is not parsed.
     *
     * @param name property name
     * @param defaultValue default value
     * @return property
     */
    public char toChar(String name, char defaultValue) {
        try {
            return parseChar(name);
        } catch (Exception e) {
            return defaultValue;
        }
    }
}
