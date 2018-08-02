/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.api.conf;

import org.datavec.api.util.ReflectionUtils;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.WritableType;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.core.JsonGenerator;
import org.nd4j.util.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.*;
import org.xml.sax.SAXException;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import java.io.*;
import java.net.URL;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

/**
 * Provides access to configuration parameters.
 *
 * <h4 id="Resources">Resources</h4>
 *
 * <p>Configurations are specified by resources. A resource contains a set of
 * name/value pairs as XML data. Each resource is named by either a 
 * <code>String</code>. If named by a <code>String</code>,
 * then the classpath is examined for a file with that name.  If named by a 
 * <code>Path</code>, then the local filesystem is examined directly, without 
 * referring to the classpath.
 *
 * <p>Unless explicitly turned off, Hadoop by default specifies two 
 * resources, loaded in-order from the classpath: <ol>
 * <li><tt><a href="{@docRoot}/../core-default.html">core-default.xml</a>
 * </tt>: Read-only defaults for hadoop.</li>
 * <li><tt>core-site.xml</tt>: Site-specific configuration for a given hadoop
 * installation.</li>
 * </ol>
 * Applications may add additional resources, which are loaded
 * subsequent to these resources in the order they are added.
 *
 * <h4 id="FinalParams">Final Parameters</h4>
 *
 * <p>Configuration parameters may be declared <i>final</i>. 
 * Once a resource declares a value final, no subsequently-loaded 
 * resource can alter that value.  
 * For example, one might define a final parameter with:
 * <tt><pre>
 *  &lt;property&gt;
 *    &lt;name&gt;dfs.client.buffer.dir&lt;/name&gt;
 *    &lt;value&gt;/tmp/hadoop/dfs/client&lt;/value&gt;
 *    <b>&lt;final&gt;true&lt;/final&gt;</b>
 *  &lt;/property&gt;</pre></tt>
 *
 * Administrators typically define parameters as final in 
 * <tt>core-site.xml</tt> for values that user applications may not alter.
 *
 * <h4 id="VariableExpansion">Variable Expansion</h4>
 *
 * <p>Value strings are first processed for <i>variable expansion</i>. The
 * available properties are:<ol>
 * <li>Other properties defined in this Configuration; and, if a name is
 * undefined here,</li>
 * <li>Properties in {@link System#getProperties()}.</li>
 * </ol>
 *
 * <p>For example, if a configuration resource contains the following property
 * definitions: 
 * <tt><pre>
 *  &lt;property&gt;
 *    &lt;name&gt;basedir&lt;/name&gt;
 *    &lt;value&gt;/user/${<i>user.name</i>}&lt;/value&gt;
 *  &lt;/property&gt;
 *
 *  &lt;property&gt;
 *    &lt;name&gt;tempdir&lt;/name&gt;
 *    &lt;value&gt;${<i>basedir</i>}/tmp&lt;/value&gt;
 *  &lt;/property&gt;</pre></tt>
 *
 * When <tt>conf.get("tempdir")</tt> is called, then <tt>${<i>basedir</i>}</tt>
 * will be resolved to another property in this Configuration, while
 * <tt>${<i>user.name</i>}</tt> would then ordinarily be resolved to the value
 * of the System property with that name.
 */
public class Configuration implements Iterable<Map.Entry<String, String>>, Writable, Serializable {
    private static final Logger LOG = LoggerFactory.getLogger(Configuration.class);

    private boolean quietmode = true;

    /**
     * List of configuration resources.
     */
    private ArrayList<Object> resources = new ArrayList<>();

    /**
     * List of configuration parameters marked <b>final</b>.
     */
    private Set<String> finalParameters = new HashSet<>();

    private boolean loadDefaults = true;

    /**
     * Configuration objects
     */
    private static final WeakHashMap<Configuration, Object> REGISTRY = new WeakHashMap<>();

    /**
     * List of default Resources. Resources are loaded in the order of the list
     * entries
     */
    private static final CopyOnWriteArrayList<String> defaultResources = new CopyOnWriteArrayList<>();

    private static final ConcurrentMap<ClassLoader, Map<String, Class<?>>> CACHE_CLASSES = new ConcurrentHashMap<>();

    /**
     * Flag to indicate if the storage of resource which updates a key needs
     * to be stored for each key
     */
    private boolean storeResource;

    /**
     * Stores the mapping of key to the resource which modifies or loads
     * the key most recently
     */
    private HashMap<String, String> updatingResource;

    static {
        //print deprecation warning if hadoop-site.xml is found in classpath
        ClassLoader cL = Thread.currentThread().getContextClassLoader();
        if (cL == null) {
            cL = Configuration.class.getClassLoader();
        }
        if (cL.getResource("hadoop-site.xml") != null) {
            LOG.warn("DEPRECATED: hadoop-site.xml found in the classpath. "
                            + "Usage of hadoop-site.xml is deprecated. Instead use core-site.xml, "
                            + "mapred-site.xml and hdfs-site.xml to override properties of "
                            + "core-default.xml, mapred-default.xml and hdfs-default.xml " + "respectively");
        }
        addDefaultResource("core-default.xml");
        addDefaultResource("core-site.xml");
    }

    private Properties properties;
    private Properties overlay;
    private transient ClassLoader classLoader;
    {
        classLoader = Thread.currentThread().getContextClassLoader();
        if (classLoader == null) {
            classLoader = Configuration.class.getClassLoader();
        }
    }



    /** A new configuration. */
    public Configuration() {
        this(true);
    }

    /** A new configuration where the behavior of reading from the default
     * resources can be turned off.
     *
     * If the parameter {@code loadDefaults} is false, the new instance
     * will not load resources from the default files.
     * @param loadDefaults specifies whether to load from the default files
     */
    public Configuration(boolean loadDefaults) {
        this.loadDefaults = loadDefaults;
        synchronized (Configuration.class) {
            REGISTRY.put(this, null);
        }
        this.storeResource = false;
    }

    /**
     * A new configuration with the same settings and additional facility for
     * storage of resource to each key which loads or updates
     * the key most recently
     * @param other the configuration from which to clone settings
     * @param storeResource flag to indicate if the storage of resource to
     * each key is to be stored
     */
    private Configuration(Configuration other, boolean storeResource) {
        this(other);
        this.loadDefaults = other.loadDefaults;
        this.storeResource = storeResource;
        if (storeResource) {
            updatingResource = new HashMap<>();
        }
    }

    /**
     * A new configuration with the same settings cloned from another.
     *
     * @param other the configuration from which to clone settings.
     */
    @SuppressWarnings("unchecked")
    public Configuration(Configuration other) {
        this.resources = (ArrayList) other.resources.clone();
        synchronized (other) {
            if (other.properties != null) {
                this.properties = (Properties) other.properties.clone();
            }

            if (other.overlay != null) {
                this.overlay = (Properties) other.overlay.clone();
            }
        }

        this.finalParameters = new HashSet<>(other.finalParameters);
        synchronized (Configuration.class) {
            REGISTRY.put(this, null);
        }
    }

    /**
     * Add a default resource. Resources are loaded in the order of the resources
     * added.
     * @param name file name. File should be present in the classpath.
     */
    public static void addDefaultResource(String name) {
        // The lock hierarchy is that we must always lock
        // instances before locking the class. Since reloadConfiguration
        // is synchronized on the instance, we must not call conf.reloadConfiguration
        // while holding a lock on Configuration.class. Otherwise we could deadlock
        // if that conf is attempting to lock the Class
        ArrayList<Configuration> toReload;
        synchronized (Configuration.class) {
            if (defaultResources.contains(name)) {
                return;
            }
            defaultResources.add(name);
            // Make a copy so we don't iterate while not holding the lock
            toReload = new ArrayList<>(REGISTRY.size());
            toReload.addAll(REGISTRY.keySet());
        }
        for (Configuration conf : toReload) {
            if (conf.loadDefaults) {
                conf.reloadConfiguration();
            }
        }
    }

    /**
     * Add a configuration resource.
     *
     * The properties of this resource will override properties of previously
     * added resources, unless they were marked <a href="#Final">final</a>.
     *
     * @param name resource to be added, the classpath is examined for a file
     *             with that name.
     */
    public void addResource(String name) {
        addResourceObject(name);
    }

    /**
     * Add a configuration resource.
     *
     * The properties of this resource will override properties of previously
     * added resources, unless they were marked <a href="#Final">final</a>.
     *
     * @param url url of the resource to be added, the local filesystem is
     *            examined directly to find the resource, without referring to
     *            the classpath.
     */
    public void addResource(URL url) {
        addResourceObject(url);
    }


    /**
     * Add a configuration resource.
     *
     * The properties of this resource will override properties of previously
     * added resources, unless they were marked <a href="#Final">final</a>.
     *
     * @param in InputStream to deserialize the object from.
     */
    public void addResource(InputStream in) {
        addResourceObject(in);
    }


    /**
     * Reload configuration from previously added resources.
     *
     * This method will clear all the configuration read from the added
     * resources, and final parameters. This will make the resources to
     * be read again before accessing the values. Values that are added
     * via set methods will overlay values read from the resources.
     */
    public synchronized void reloadConfiguration() {
        properties = null; // trigger reload
        finalParameters.clear(); // clear site-limits
    }

    private synchronized void addResourceObject(Object resource) {
        resources.add(resource); // add to resources
        reloadConfiguration();
    }

    private static Pattern varPat = Pattern.compile("\\$\\{[^\\}\\$\u0020]+\\}");

    private String substituteVars(String expr) {
        if (expr == null) {
            return null;
        }
        Matcher match = varPat.matcher("");
        String eval = expr;
        int MAX_SUBST = 20;
        for (int s = 0; s < MAX_SUBST; s++) {
            match.reset(eval);
            if (!match.find()) {
                return eval;
            }
            String var = match.group();
            var = var.substring(2, var.length() - 1); // remove ${ .. }
            String val = null;
            try {
                val = System.getProperty(var);
            } catch (SecurityException se) {
                LOG.warn("Unexpected SecurityException in Configuration", se);
            }
            if (val == null) {
                val = getRaw(var);
            }
            if (val == null) {
                return eval; // return literal ${var}: var is unbound
            }
            // substitute
            eval = eval.substring(0, match.start()) + val + eval.substring(match.end());
        }
        throw new IllegalStateException("Variable substitution depth too large: " + MAX_SUBST + " " + expr);
    }

    /**
     * Get the value of the <code>name</code> property, <code>null</code> if
     * no such property exists.
     *
     * Values are processed for <a href="#VariableExpansion">variable expansion</a>
     * before being returned.
     *
     * @param name the property name.
     * @return the value of the <code>name</code> property,
     *         or null if no such property exists.
     */
    public String get(String name) {
        return substituteVars(getProps().getProperty(name));
    }

    /**
     * Get the value of the <code>name</code> property, without doing
     * <a href="#VariableExpansion">variable expansion</a>.
     *
     * @param name the property name.
     * @return the value of the <code>name</code> property,
     *         or null if no such property exists.
     */
    public String getRaw(String name) {
        return getProps().getProperty(name);
    }

    /**
     * Get the char value of the <code>name</code> property, <code>null</code> if
     * no such property exists.
     *
     * Values are processed for <a href="#VariableExpansion">variable expansion</a>
     * before being returned.
     *
     * @param name the property name.
     * @return the value of the <code>name</code> property,
     *         or null if no such property exists.
     */
    public char getChar(String name) {
        return getProps().getProperty(name).charAt(0);
    }

    /**
     * Get the char value of the <code>name</code> property, <code>null</code> if
     * no such property exists.
     *
     * Values are processed for <a href="#VariableExpansion">variable expansion</a>
     * before being returned.
     *
     * @param name the property name.
     * @return the value of the <code>name</code> property,
     *         or null if no such property exists.
     */
    public char getChar(String name, char defaultValue) {
        return getProps().getProperty(name, String.valueOf(defaultValue)).charAt(0);
    }

    /**
     * Set the <code>value</code> of the <code>name</code> property.
     *
     * @param name property name.
     * @param value property value.
     */
    public void set(String name, String value) {
        getOverlay().setProperty(name, value);
        getProps().setProperty(name, value);
    }

    /**
     * Sets a property if it is currently unset.
     * @param name the property name
     * @param value the new value
     */
    public void setIfUnset(String name, String value) {
        if (get(name) == null) {
            set(name, value);
        }
    }

    private synchronized Properties getOverlay() {
        if (overlay == null) {
            overlay = new Properties();
        }
        return overlay;
    }

    /**
     * Get the value of the <code>name</code> property. If no such property
     * exists, then <code>defaultValue</code> is returned.
     *
     * @param name property name.
     * @param defaultValue default value.
     * @return property value, or <code>defaultValue</code> if the property
     *         doesn't exist.
     */
    public String get(String name, String defaultValue) {
        return substituteVars(getProps().getProperty(name, defaultValue));
    }

    /**
     * Get the value of the <code>name</code> property as an <code>int</code>.
     *
     * If no such property exists, or if the specified value is not a valid
     * <code>int</code>, then <code>defaultValue</code> is returned.
     *
     * @param name property name.
     * @param defaultValue default value.
     * @return property value as an <code>int</code>,
     *         or <code>defaultValue</code>.
     */
    public int getInt(String name, int defaultValue) {
        String valueString = get(name);
        if (valueString == null)
            return defaultValue;
        try {
            String hexString = getHexDigits(valueString);
            if (hexString != null) {
                return Integer.parseInt(hexString, 16);
            }
            return Integer.parseInt(valueString);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    /**
     * Set the value of the <code>name</code> property to an <code>int</code>.
     *
     * @param name property name.
     * @param value <code>int</code> value of the property.
     */
    public void setInt(String name, int value) {
        set(name, Integer.toString(value));
    }


    /**
     * Get the value of the <code>name</code> property as a <code>long</code>.
     * If no such property is specified, or if the specified value is not a valid
     * <code>long</code>, then <code>defaultValue</code> is returned.
     *
     * @param name property name.
     * @param defaultValue default value.
     * @return property value as a <code>long</code>,
     *         or <code>defaultValue</code>.
     */
    public long getLong(String name, long defaultValue) {
        String valueString = get(name);
        if (valueString == null)
            return defaultValue;
        try {
            String hexString = getHexDigits(valueString);
            if (hexString != null) {
                return Long.parseLong(hexString, 16);
            }
            return Long.parseLong(valueString);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    private String getHexDigits(String value) {
        boolean negative = false;
        String str = value;
        String hexString;
        if (value.startsWith("-")) {
            negative = true;
            str = value.substring(1);
        }
        if (str.startsWith("0x") || str.startsWith("0X")) {
            hexString = str.substring(2);
            if (negative) {
                hexString = "-" + hexString;
            }
            return hexString;
        }
        return null;
    }

    /**
     * Set the value of the <code>name</code> property to a <code>long</code>.
     *
     * @param name property name.
     * @param value <code>long</code> value of the property.
     */
    public void setLong(String name, long value) {
        set(name, Long.toString(value));
    }

    /**
     * Get the value of the <code>name</code> property as a <code>float</code>.
     * If no such property is specified, or if the specified value is not a valid
     * <code>float</code>, then <code>defaultValue</code> is returned.
     *
     * @param name property name.
     * @param defaultValue default value.
     * @return property value as a <code>float</code>,
     *         or <code>defaultValue</code>.
     */
    public float getFloat(String name, float defaultValue) {
        String valueString = get(name);
        if (valueString == null)
            return defaultValue;
        try {
            return Float.parseFloat(valueString);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    /**
     * Set the value of the <code>name</code> property to a <code>float</code>.
     *
     * @param name property name.
     * @param value property value.
     */
    public void setFloat(String name, float value) {
        set(name, Float.toString(value));
    }

    /**
     * Get the value of the <code>name</code> property as a <code>boolean</code>.
     * If no such property is specified, or if the specified value is not a valid
     * <code>boolean</code>, then <code>defaultValue</code> is returned.
     *
     * @param name property name.
     * @param defaultValue default value.
     * @return property value as a <code>boolean</code>,
     *         or <code>defaultValue</code>.
     */
    public boolean getBoolean(String name, boolean defaultValue) {
        String valueString = get(name);
        return "true".equals(valueString) || !"false".equals(valueString) && defaultValue;
    }

    /**
     * Set the value of the <code>name</code> property to a <code>boolean</code>.
     *
     * @param name property name.
     * @param value <code>boolean</code> value of the property.
     */
    public void setBoolean(String name, boolean value) {
        set(name, Boolean.toString(value));
    }

    /**
     * Set the given property, if it is currently unset.
     * @param name property name
     * @param value new value
     */
    public void setBooleanIfUnset(String name, boolean value) {
        setIfUnset(name, Boolean.toString(value));
    }

    /**
     * Get the value of the <code>name</code> property as a <ocde>Pattern</code>.
     * If no such property is specified, or if the specified value is not a valid
     * <code>Pattern</code>, then <code>DefaultValue</code> is returned.
     *
     * @param name property name
     * @param defaultValue default value
     * @return property value as a compiled Pattern, or defaultValue
     */
    public Pattern getPattern(String name, Pattern defaultValue) {
        String valString = get(name);
        if (null == valString || "".equals(valString)) {
            return defaultValue;
        }
        try {
            return Pattern.compile(valString);
        } catch (PatternSyntaxException pse) {
            LOG.warn("Regular expression '" + valString + "' for property '" + name + "' not valid. Using default",
                            pse);
            return defaultValue;
        }
    }

    /**
     * Set the given property to <code>Pattern</code>.
     * If the pattern is passed as null, sets the empty pattern which results in
     * further calls to getPattern(...) returning the default value.
     *
     * @param name property name
     * @param pattern new value
     */
    public void setPattern(String name, Pattern pattern) {
        if (null == pattern) {
            set(name, null);
        } else {
            set(name, pattern.pattern());
        }
    }

    @Override
    public void write(DataOutput out) throws IOException {

    }

    @Override
    public void readFields(DataInput in) throws IOException {

    }

    /**
     * A class that represents a set of positive integer ranges. It parses
     * strings of the form: "2-3,5,7-" where ranges are separated by comma and
     * the lower/upper bounds are separated by dash. Either the lower or upper
     * bound may be omitted meaning all values up to or over. So the string
     * above means 2, 3, 5, and 7, 8, 9, ...
     */
    public static class IntegerRanges {
        private static class Range {
            int start;
            int end;
        }

        List<Range> ranges = new ArrayList<Range>();

        public IntegerRanges() {}

        public IntegerRanges(String newValue) {
            StringTokenizer itr = new StringTokenizer(newValue, ",");
            while (itr.hasMoreTokens()) {
                String rng = itr.nextToken().trim();
                String[] parts = rng.split("-", 3);
                if (parts.length < 1 || parts.length > 2) {
                    throw new IllegalArgumentException("integer range badly formed: " + rng);
                }
                Range r = new Range();
                r.start = convertToInt(parts[0], 0);
                if (parts.length == 2) {
                    r.end = convertToInt(parts[1], Integer.MAX_VALUE);
                } else {
                    r.end = r.start;
                }
                if (r.start > r.end) {
                    throw new IllegalArgumentException("IntegerRange from " + r.start + " to " + r.end + " is invalid");
                }
                ranges.add(r);
            }
        }

        /**
         * Convert a string to an int treating empty strings as the default value.
         * @param value the string value
         * @param defaultValue the value for if the string is empty
         * @return the desired integer
         */
        private static int convertToInt(String value, int defaultValue) {
            String trim = value.trim();
            if (trim.length() == 0) {
                return defaultValue;
            }
            return Integer.parseInt(trim);
        }

        /**
         * Is the given value in the set of ranges
         * @param value the value to check
         * @return is the value in the ranges?
         */
        public boolean isIncluded(int value) {
            for (Range r : ranges) {
                if (r.start <= value && value <= r.end) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public String toString() {
            StringBuilder result = new StringBuilder();
            boolean first = true;
            for (Range r : ranges) {
                if (first) {
                    first = false;
                } else {
                    result.append(',');
                }
                result.append(r.start);
                result.append('-');
                result.append(r.end);
            }
            return result.toString();
        }
    }

    /**
     * Parse the given attribute as a set of integer ranges
     * @param name the attribute name
     * @param defaultValue the default value if it is not set
     * @return a new set of ranges from the configured value
     */
    public IntegerRanges getRange(String name, String defaultValue) {
        return new IntegerRanges(get(name, defaultValue));
    }

    /**
     * Get the comma delimited values of the <code>name</code> property as
     * a collection of <code>String</code>s.
     * If no such property is specified then empty collection is returned.
     * <p>
     * This is an optimized version of {@link #getStrings(String)}
     *
     * @param name property name.
     * @return property value as a collection of <code>String</code>s.
     */
    public Collection<String> getStringCollection(String name) {
        String valueString = get(name);
        return StringUtils.getStringCollection(valueString);
    }

    /**
     * Get the comma delimited values of the <code>name</code> property as
     * an array of <code>String</code>s.
     * If no such property is specified then <code>null</code> is returned.
     *
     * @param name property name.
     * @return property value as an array of <code>String</code>s,
     *         or <code>null</code>.
     */
    public String[] getStrings(String name) {
        String valueString = get(name);
        return StringUtils.getStrings(valueString);
    }

    /**
     * Get the comma delimited values of the <code>name</code> property as
     * an array of <code>String</code>s.
     * If no such property is specified then default value is returned.
     *
     * @param name property name.
     * @param defaultValue The default value
     * @return property value as an array of <code>String</code>s,
     *         or default value.
     */
    public String[] getStrings(String name, String... defaultValue) {
        String valueString = get(name);
        if (valueString == null) {
            return defaultValue;
        } else {
            return StringUtils.getStrings(valueString);
        }
    }

    /**
     * Get the comma delimited values of the <code>name</code> property as
     * a collection of <code>String</code>s, trimmed of the leading and trailing whitespace.
     * If no such property is specified then empty <code>Collection</code> is returned.
     *
     * @param name property name.
     * @return property value as a collection of <code>String</code>s, or empty <code>Collection</code>
     */
    public Collection<String> getTrimmedStringCollection(String name) {
        String valueString = get(name);
        if (null == valueString) {
            return Collections.emptyList();
        }
        return StringUtils.getTrimmedStringCollection(valueString);
    }

    /**
     * Get the comma delimited values of the <code>name</code> property as
     * an array of <code>String</code>s, trimmed of the leading and trailing whitespace.
     * If no such property is specified then an empty array is returned.
     *
     * @param name property name.
     * @return property value as an array of trimmed <code>String</code>s,
     *         or empty array.
     */
    public String[] getTrimmedStrings(String name) {
        String valueString = get(name);
        return StringUtils.getTrimmedStrings(valueString);
    }

    /**
     * Get the comma delimited values of the <code>name</code> property as
     * an array of <code>String</code>s, trimmed of the leading and trailing whitespace.
     * If no such property is specified then default value is returned.
     *
     * @param name property name.
     * @param defaultValue The default value
     * @return property value as an array of trimmed <code>String</code>s,
     *         or default value.
     */
    public String[] getTrimmedStrings(String name, String... defaultValue) {
        String valueString = get(name);
        if (null == valueString) {
            return defaultValue;
        } else {
            return StringUtils.getTrimmedStrings(valueString);
        }
    }

    /**
     * Set the array of string values for the <code>name</code> property as
     * as comma delimited values.
     *
     * @param name property name.
     * @param values The values
     */
    public void setStrings(String name, String... values) {
        set(name, StringUtils.arrayToString(values));
    }

    /**
     * Load a class by name.
     *
     * @param name the class name.
     * @return the class object.
     * @throws ClassNotFoundException if the class is not found.
     */
    public Class<?> getClassByName(String name) throws ClassNotFoundException {
        Map<String, Class<?>> map = CACHE_CLASSES.get(classLoader);
        if (map == null) {
            Map<String, Class<?>> newMap = new ConcurrentHashMap<>();
            map = CACHE_CLASSES.putIfAbsent(classLoader, newMap);
            if (map == null) {
                map = newMap;
            }
        }

        Class clazz = map.get(name);
        if (clazz == null) {
            clazz = Class.forName(name, true, classLoader);
            if (clazz != null) {
                map.put(name, clazz);
            }
        }

        return clazz;
    }

    /**
     * Get the value of the <code>name</code> property
     * as an array of <code>Class</code>.
     * The value of the property specifies a list of comma separated class names.
     * If no such property is specified, then <code>defaultValue</code> is
     * returned.
     *
     * @param name the property name.
     * @param defaultValue default value.
     * @return property value as a <code>Class[]</code>,
     *         or <code>defaultValue</code>.
     */
    public Class<?>[] getClasses(String name, Class<?>... defaultValue) {
        String[] classnames = getStrings(name);
        if (classnames == null)
            return defaultValue;
        try {
            Class<?>[] classes = new Class<?>[classnames.length];
            for (int i = 0; i < classnames.length; i++) {
                classes[i] = getClassByName(classnames[i]);
            }
            return classes;
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Get the value of the <code>name</code> property as a <code>Class</code>.
     * If no such property is specified, then <code>defaultValue</code> is
     * returned.
     *
     * @param name the class name.
     * @param defaultValue default value.
     * @return property value as a <code>Class</code>,
     *         or <code>defaultValue</code>.
     */
    public Class<?> getClass(String name, Class<?> defaultValue) {
        String valueString = get(name);
        if (valueString == null)
            return defaultValue;
        try {
            return getClassByName(valueString);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Get the value of the <code>name</code> property as a <code>Class</code>
     * implementing the interface specified by <code>xface</code>.
     *
     * If no such property is specified, then <code>defaultValue</code> is
     * returned.
     *
     * An exception is thrown if the returned class does not implement the named
     * interface.
     *
     * @param name the class name.
     * @param defaultValue default value.
     * @param xface the interface implemented by the named class.
     * @return property value as a <code>Class</code>,
     *         or <code>defaultValue</code>.
     */
    public <U> Class<? extends U> getClass(String name, Class<? extends U> defaultValue, Class<U> xface) {
        try {
            Class<?> theClass = getClass(name, defaultValue);
            if (theClass != null && !xface.isAssignableFrom(theClass))
                throw new RuntimeException(theClass + " not " + xface.getName());
            else if (theClass != null)
                return theClass.asSubclass(xface);
            else
                return null;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Get the value of the <code>name</code> property as a <code>List</code>
     * of objects implementing the interface specified by <code>xface</code>.
     *
     * An exception is thrown if any of the classes does not exist, or if it does
     * not implement the named interface.
     *
     * @param name the property name.
     * @param xface the interface implemented by the classes named by
     *        <code>name</code>.
     * @return a <code>List</code> of objects implementing <code>xface</code>.
     */
    @SuppressWarnings("unchecked")
    public <U> List<U> getInstances(String name, Class<U> xface) {
        List<U> ret = new ArrayList<>();
        Class<?>[] classes = getClasses(name);
        for (Class<?> cl : classes) {
            if (!xface.isAssignableFrom(cl)) {
                throw new RuntimeException(cl + " does not implement " + xface);
            }
            ret.add((U) ReflectionUtils.newInstance(cl, this));
        }
        return ret;
    }

    /**
     * Set the value of the <code>name</code> property to the name of a
     * <code>theClass</code> implementing the given interface <code>xface</code>.
     *
     * An exception is thrown if <code>theClass</code> does not implement the
     * interface <code>xface</code>.
     *
     * @param name property name.
     * @param theClass property value.
     * @param xface the interface implemented by the named class.
     */
    public void setClass(String name, Class<?> theClass, Class<?> xface) {
        if (!xface.isAssignableFrom(theClass))
            throw new RuntimeException(theClass + " not " + xface.getName());
        set(name, theClass.getName());
    }



    /**
     * Get a local file name under a directory named in <i>dirsProp</i> with
     * the given <i>path</i>.  If <i>dirsProp</i> contains multiple directories,
     * then one is chosen based on <i>path</i>'s hash code.  If the selected
     * directory does not exist, an attempt is made to create it.
     *
     * @param dirsProp directory in which to locate the file.
     * @param path file-path.
     * @return local file under the directory with the given path.
     */
    public File getFile(String dirsProp, String path) throws IOException {
        String[] dirs = getStrings(dirsProp);
        int hashCode = path.hashCode();
        for (int i = 0; i < dirs.length; i++) { // try each local dir
            int index = (hashCode + i & Integer.MAX_VALUE) % dirs.length;
            File file = new File(dirs[index], path);
            File dir = file.getParentFile();
            if (dir.exists() || dir.mkdirs()) {
                return file;
            }
        }
        throw new IOException("No valid local directories in property: " + dirsProp);
    }

    /**
     * Get the {@link URL} for the named resource.
     *
     * @param name resource name.
     * @return the url for the named resource.
     */
    public URL getResource(String name) {
        return classLoader.getResource(name);
    }

    /**
     * Get an input stream attached to the configuration resource with the
     * given <code>name</code>.
     *
     * @param name configuration resource name.
     * @return an input stream attached to the resource.
     */
    public InputStream getConfResourceAsInputStream(String name) {
        try {
            URL url = getResource(name);

            if (url == null) {
                LOG.info(name + " not found");
                return null;
            } else {
                LOG.info("found resource " + name + " at " + url);
            }

            return url.openStream();
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Get a {@link Reader} attached to the configuration resource with the
     * given <code>name</code>.
     *
     * @param name configuration resource name.
     * @return a reader attached to the resource.
     */
    public Reader getConfResourceAsReader(String name) {
        try {
            URL url = getResource(name);

            if (url == null) {
                LOG.info(name + " not found");
                return null;
            } else {
                LOG.info("found resource " + name + " at " + url);
            }

            return new InputStreamReader(url.openStream());
        } catch (Exception e) {
            return null;
        }
    }

    private synchronized Properties getProps() {
        if (properties == null) {
            properties = new Properties();
            loadResources(properties, resources, quietmode);
            if (overlay != null) {
                properties.putAll(overlay);
                if (storeResource) {
                    for (Map.Entry<Object, Object> item : overlay.entrySet()) {
                        updatingResource.put((String) item.getKey(), "Unknown");
                    }
                }
            }
        }
        return properties;
    }

    /**
     * Return the number of keys in the configuration.
     *
     * @return number of keys in the configuration.
     */
    public int size() {
        return getProps().size();
    }

    /**
     * Clears all keys from the configuration.
     */
    public void clear() {
        getProps().clear();
        getOverlay().clear();
    }

    /**
     * Get an {@link Iterator} to go through the list of <code>String</code>
     * key-value pairs in the configuration.
     *
     * @return an iterator over the entries.
     */
    public Iterator<Map.Entry<String, String>> iterator() {
        // Get a copy of just the string to string pairs. After the old object
        // methods that allow non-strings to be put into configurations are removed,
        // we could replace properties with a Map<String,String> and get rid of this
        // code.
        Map<String, String> result = new HashMap<>();
        for (Map.Entry<Object, Object> item : getProps().entrySet()) {
            if (item.getKey() instanceof String && item.getValue() instanceof String) {
                result.put((String) item.getKey(), (String) item.getValue());
            }
        }
        return result.entrySet().iterator();
    }

    private void loadResources(Properties properties, ArrayList resources, boolean quiet) {
        if (loadDefaults) {
            // To avoid addResource causing a ConcurrentModificationException
            ArrayList<String> toLoad;
            synchronized (Configuration.class) {
                toLoad = new ArrayList<>(defaultResources);
            }
            for (String resource : toLoad) {
                loadResource(properties, resource, quiet);
            }

            //support the hadoop-site.xml as a deprecated case
            if (getResource("hadoop-site.xml") != null) {
                loadResource(properties, "hadoop-site.xml", quiet);
            }
        }

        for (Object resource : resources) {
            loadResource(properties, resource, quiet);
        }
    }

    private void loadResource(Properties properties, Object name, boolean quiet) {
        try {
            DocumentBuilderFactory docBuilderFactory = DocumentBuilderFactory.newInstance();
            //ignore all comments inside the xml file
            docBuilderFactory.setIgnoringComments(true);

            //allow includes in the xml file
            docBuilderFactory.setNamespaceAware(true);
            try {
                docBuilderFactory.setXIncludeAware(true);
            } catch (UnsupportedOperationException e) {
                LOG.error("Failed to set setXIncludeAware(true) for parser " + docBuilderFactory + ":" + e, e);
            }
            DocumentBuilder builder = docBuilderFactory.newDocumentBuilder();
            Document doc = null;
            Element root = null;

            if (name instanceof URL) { // an URL resource
                URL url = (URL) name;
                if (url != null) {
                    if (!quiet) {
                        LOG.info("parsing " + url);
                    }
                    doc = builder.parse(url.toString());
                }
            } else if (name instanceof String) { // a CLASSPATH resource
                URL url = getResource((String) name);
                if (url != null) {
                    if (!quiet) {
                        LOG.info("parsing " + url);
                    }
                    doc = builder.parse(url.toString());
                }
            } else if (name instanceof InputStream) {
                try {
                    doc = builder.parse((InputStream) name);
                } finally {
                    ((InputStream) name).close();
                }
            } else if (name instanceof Element) {
                root = (Element) name;
            }

            if (doc == null && root == null) {
                if (quiet)
                    return;
                throw new RuntimeException(name + " not found");
            }

            if (root == null) {
                root = doc.getDocumentElement();
            }
            if (!"configuration".equals(root.getTagName()))
                LOG.error("bad conf file: top-level element not <configuration>");
            NodeList props = root.getChildNodes();
            for (int i = 0; i < props.getLength(); i++) {
                Node propNode = props.item(i);
                if (!(propNode instanceof Element))
                    continue;
                Element prop = (Element) propNode;
                if ("configuration".equals(prop.getTagName())) {
                    loadResource(properties, prop, quiet);
                    continue;
                }
                if (!"property".equals(prop.getTagName()))
                    LOG.warn("bad conf file: element not <property>");
                NodeList fields = prop.getChildNodes();
                String attr = null;
                String value = null;
                boolean finalParameter = false;
                for (int j = 0; j < fields.getLength(); j++) {
                    Node fieldNode = fields.item(j);
                    if (!(fieldNode instanceof Element))
                        continue;
                    Element field = (Element) fieldNode;
                    if ("name".equals(field.getTagName()) && field.hasChildNodes())
                        attr = ((Text) field.getFirstChild()).getData().trim();
                    if ("value".equals(field.getTagName()) && field.hasChildNodes())
                        value = ((Text) field.getFirstChild()).getData();
                    if ("final".equals(field.getTagName()) && field.hasChildNodes())
                        finalParameter = "true".equals(((Text) field.getFirstChild()).getData());
                }

                // Ignore this parameter if it has already been marked as 'final'
                if (attr != null && value != null) {
                    if (!finalParameters.contains(attr)) {
                        properties.setProperty(attr, value);
                        if (storeResource) {
                            updatingResource.put(attr, name.toString());
                        }
                        if (finalParameter)
                            finalParameters.add(attr);
                    } else {
                        LOG.warn(name + ":a attempt to override final parameter: " + attr + ";  Ignoring.");
                    }
                }
            }

        } catch (IOException | ParserConfigurationException | SAXException | DOMException e) {
            LOG.error("error parsing conf file: " + e);
            throw new RuntimeException(e);
        }
    }

    /**
     * Write out the non-default properties in this configuration to the give
     * {@link OutputStream}.
     *
     * @param out the output stream to write to.
     */
    public void writeXml(OutputStream out) throws IOException {
        Properties properties = getProps();
        try {
            Document doc = DocumentBuilderFactory.newInstance().newDocumentBuilder().newDocument();
            Element conf = doc.createElement("configuration");
            doc.appendChild(conf);
            conf.appendChild(doc.createTextNode("\n"));
            for (Enumeration e = properties.keys(); e.hasMoreElements();) {
                String name = (String) e.nextElement();
                Object object = properties.get(name);
                String value;
                if (object instanceof String) {
                    value = (String) object;
                } else {
                    continue;
                }
                Element propNode = doc.createElement("property");
                conf.appendChild(propNode);

                Element nameNode = doc.createElement("name");
                nameNode.appendChild(doc.createTextNode(name));
                propNode.appendChild(nameNode);

                Element valueNode = doc.createElement("value");
                valueNode.appendChild(doc.createTextNode(value));
                propNode.appendChild(valueNode);

                conf.appendChild(doc.createTextNode("\n"));
            }

            DOMSource source = new DOMSource(doc);
            StreamResult result = new StreamResult(out);
            TransformerFactory transFactory = TransformerFactory.newInstance();
            Transformer transformer = transFactory.newTransformer();
            transformer.transform(source, result);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     *  Writes out all the parameters and their properties (final and resource) to
     *  the given {@link Writer}
     *  The format of the output would be
     *  { "properties" : [ {key1,value1,key1.isFinal,key1.resource}, {key2,value2,
     *  key2.isFinal,key2.resource}... ] }
     *  It does not output the parameters of the configuration object which is
     *  loaded from an input stream.
     * @param out the Writer to write to
     * @throws IOException
     */
    public static void dumpConfiguration(Configuration conf, Writer out) throws IOException {
        Configuration config = new Configuration(conf, true);
        config.reloadConfiguration();
        JsonFactory dumpFactory = new JsonFactory();
        JsonGenerator dumpGenerator = dumpFactory.createGenerator(out);
        dumpGenerator.writeStartObject();
        dumpGenerator.writeFieldName("properties");
        dumpGenerator.writeStartArray();
        dumpGenerator.flush();
        for (Map.Entry<Object, Object> item : config.getProps().entrySet()) {
            dumpGenerator.writeStartObject();
            dumpGenerator.writeStringField("key", (String) item.getKey());
            dumpGenerator.writeStringField("value", config.get((String) item.getKey()));
            dumpGenerator.writeBooleanField("isFinal", config.finalParameters.contains(item.getKey()));
            dumpGenerator.writeStringField("resource", config.updatingResource.get(item.getKey()));
            dumpGenerator.writeEndObject();
        }
        dumpGenerator.writeEndArray();
        dumpGenerator.writeEndObject();
        dumpGenerator.flush();
    }

    /**
     * Get the {@link ClassLoader} for this job.
     *
     * @return the correct class loader.
     */
    public ClassLoader getClassLoader() {
        return classLoader;
    }

    /**
     * Set the class loader that will be used to load the various objects.
     *
     * @param classLoader the new class loader.
     */
    public void setClassLoader(ClassLoader classLoader) {
        this.classLoader = classLoader;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Configuration: ");
        if (loadDefaults) {
            synchronized (Configuration.class) {
                toString(defaultResources, sb);
            }
            if (resources.size() > 0) {
                sb.append(", ");
            }
        }
        toString(resources, sb);
        return sb.toString();
    }

    private void toString(List resources, StringBuilder sb) {
        ListIterator i = resources.listIterator();
        while (i.hasNext()) {
            if (i.nextIndex() != 0) {
                sb.append(", ");
            }
            sb.append(i.next());
        }
    }

    /**
     * Set the quietness-mode.
     *
     * In the quiet-mode, error and informational messages might not be logged.
     *
     * @param quietmode <code>true</code> to set quiet-mode on, <code>false</code>
     *              to turn it off.
     */
    public synchronized void setQuietMode(boolean quietmode) {
        this.quietmode = quietmode;
    }

    /** For debugging.  List non-default properties to the terminal and exit. */
    public static void main(String[] args) throws Exception {
        new Configuration().writeXml(System.out);
    }


    @Override
    public double toDouble() {
        throw new UnsupportedOperationException();
    }

    @Override
    public float toFloat() {
        throw new UnsupportedOperationException();
    }

    @Override
    public int toInt() {
        throw new UnsupportedOperationException();
    }

    @Override
    public long toLong() {
        throw new UnsupportedOperationException();
    }

    @Override
    public WritableType getType() {
        throw new UnsupportedOperationException();
    }

    @Override
    public void writeType(DataOutput out) throws IOException {
        throw new UnsupportedOperationException();
    }
}
