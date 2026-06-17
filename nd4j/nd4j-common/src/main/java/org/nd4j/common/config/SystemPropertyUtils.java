/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.common.config;

/**
 * Static utility methods for reading typed system properties with defaults.
 * Centralizes the repeated pattern of:
 * <pre>
 *   String val = System.getProperty(key);
 *   if (val != null) { result = Boolean.parseBoolean(val); }
 * </pre>
 */
public class SystemPropertyUtils {

    private SystemPropertyUtils() {}

    public static boolean getBooleanProperty(String key, boolean defaultValue) {
        String val = System.getProperty(key);
        return val != null ? Boolean.parseBoolean(val) : defaultValue;
    }

    public static int getIntProperty(String key, int defaultValue) {
        String val = System.getProperty(key);
        if (val == null) return defaultValue;
        try {
            return Integer.parseInt(val);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    public static long getLongProperty(String key, long defaultValue) {
        String val = System.getProperty(key);
        if (val == null) return defaultValue;
        try {
            return Long.parseLong(val);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    public static double getDoubleProperty(String key, double defaultValue) {
        String val = System.getProperty(key);
        if (val == null) return defaultValue;
        try {
            return Double.parseDouble(val);
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }

    public static String getStringProperty(String key, String defaultValue) {
        return System.getProperty(key, defaultValue);
    }

    public static boolean hasProperty(String key) {
        return System.getProperty(key) != null;
    }
}
