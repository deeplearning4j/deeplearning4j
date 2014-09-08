package org.nd4j.jdbc.driverfinder;

import org.reflections.Reflections;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Modifier;
import java.sql.Driver;
import java.util.HashSet;
import java.util.Properties;
import java.util.Set;

/**
 * JDBC Driver finder
 * @author Adam Gibson
 */
public class DriverFinder {

    private static Class<? extends Driver> clazz;
    private static Driver driver;
    public final static String ND4j_JDBC_PROPERTIES = "nd4j.jdbc.properties";
    public final static String JDBC_KEY = "jdbc.driver";

    public static Driver getDriver() {
        if(driver == null) {
            if(clazz == null)
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
        Reflections r = new Reflections();
        Set<Class<? extends Driver>> clazzes = new HashSet<>(r.getSubTypesOf(Driver.class));
        if(clazzes.isEmpty()) {
            throw new IllegalStateException("No org.nd4j.jdbc drivers found.");
        }
        else if(clazzes.size() != 1) {
            Set<Class<? extends Driver>> remove = new HashSet<>();
            for(Class<? extends Driver> clazz : clazzes) {
                if(Modifier.isAbstract(clazz.getModifiers())) {
                    remove.add(clazz);
                }
                else if(Modifier.isInterface(clazz.getModifiers())) {
                    remove.add(clazz);
                }
            }

            clazzes.removeAll(remove);
            if(clazzes.size() != 1) {
                  InputStream i = DriverFinder.class.getResourceAsStream("/" + ND4j_JDBC_PROPERTIES);
                  if(i == null)
                      throw new IllegalStateException("Only one jdbc driver allowed on the class path");
                else {
                      Properties props = new Properties();
                      try {
                          props.load(i);
                      } catch (IOException e) {
                          throw new RuntimeException(e);
                      }

                      String clazz = props.getProperty(JDBC_KEY);
                      if(clazz == null)
                          throw new IllegalStateException("Unable to find jdbc driver. Please specify a " + ND4j_JDBC_PROPERTIES + " with the key " + JDBC_KEY);
                      try {
                          DriverFinder.clazz = (Class<? extends Driver>) Class.forName(clazz);
                      } catch (ClassNotFoundException e) {
                          throw new IllegalStateException("Unable to find jdbc driver. Please specify a " + ND4j_JDBC_PROPERTIES + " with the key " + JDBC_KEY);

                      }

                  }

            }
        }
    }


}
