package org.nd4j.context;

import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;
import java.util.Properties;

/**
 * Holds properties for nd4j to be used
 * across different modules
 *
 * @author Adam Gibson
 */
public class Nd4jContext implements Serializable {
    private  Properties conf;
    private static Nd4jContext INSTANCE;

    private Nd4jContext() {}

    public static Nd4jContext getInstance() {
        if(INSTANCE == null)
            INSTANCE = new Nd4jContext();
        return INSTANCE;
    }

    /**
     * Load the properties
     * from an input stream
     * @param inputStream
     */
    public void updateProperties(InputStream inputStream) {
        if(conf == null) {
            conf = new Properties();
            conf.putAll(System.getProperties());
        }

        try {
            conf.load(inputStream);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Get the configuration for nd4j
     * @return
     */
    public  Properties getConf() {
        if(conf == null) {
            conf = new Properties();
            conf.putAll(System.getProperties());
        }

        return conf;
    }
}
