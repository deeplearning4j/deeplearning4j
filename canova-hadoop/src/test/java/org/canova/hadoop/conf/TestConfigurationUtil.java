package org.canova.hadoop.conf;

import static org.junit.Assert.*;

import org.apache.hadoop.conf.Configuration;
import org.junit.Test;

public class TestConfigurationUtil {

	@Test
	public void testLoadHadoopConfFiles() {
		
		// this would come from the properties file
		String confPath = "src/test/resources/conf/example_conf/";
		
		Configuration conf = ConfigurationUtil.generateConfig(confPath);
		
		System.out.println( " works? " + conf.get("fs.default.name") );
		
		
	}

}
