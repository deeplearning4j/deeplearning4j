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

package org.nd4j.jdbc.driverfinder;

import java.io.IOException;
import java.io.InputStream;
import java.sql.Driver;
import java.util.HashSet;
import java.util.Properties;
import java.util.ServiceLoader;
import java.util.Set;

/**
 * JDBC Driver finder
 *
 * @author Adam Gibson
 */
public class DriverFinder {

    public final static String ND4j_JDBC_PROPERTIES = "nd4j.jdbc.properties";
    public final static String JDBC_KEY = "jdbc.driver";
    private static Class<? extends Driver> clazz;
    private static Driver driver;

    public static Driver getDriver() {
        if (driver == null) {
            if (clazz == null)
                discoverDriverClazz();
            try {
                driver = clazz.newInstance();
            } catch (InstantiationException e) {
                e.printStackTrace();
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }
        }
        return driver;
    }


    private static void discoverDriverClazz() {
        //All JDBC4 compliant drivers support ServiceLoader mechanism for discovery - https://stackoverflow.com/a/18297412
        ServiceLoader<Driver> drivers = ServiceLoader.load(Driver.class);
        Set<Class<? extends Driver>> driverClasses = new HashSet<>();
        for(Driver d : drivers){
            driverClasses.add(d.getClass());
        }

        if(driverClasses.isEmpty()){
            throw new IllegalStateException("No org.nd4j.jdbc drivers found on classpath via ServiceLoader");
        }

        if(driverClasses.size() != 1) {
            InputStream i = DriverFinder.class.getResourceAsStream("/" + ND4j_JDBC_PROPERTIES);
            if (i == null)
                throw new IllegalStateException("Only one jdbc driver allowed on the class path");
            else {
                Properties props = new Properties();
                try {
                    props.load(i);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }

                String clazz = props.getProperty(JDBC_KEY);
                if (clazz == null)
                    throw new IllegalStateException("Unable to find jdbc driver. Please specify a "
                            + ND4j_JDBC_PROPERTIES + " with the key " + JDBC_KEY);
                try {
                    DriverFinder.clazz = (Class<? extends Driver>) Class.forName(clazz);
                } catch (ClassNotFoundException e) {
                    throw new IllegalStateException("Unable to find jdbc driver. Please specify a "
                            + ND4j_JDBC_PROPERTIES + " with the key " + JDBC_KEY);
                }
            }
        }
    }
}
