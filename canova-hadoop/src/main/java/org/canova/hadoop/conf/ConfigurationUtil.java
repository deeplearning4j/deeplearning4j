package org.canova.hadoop.conf;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

/**
 * Notes
 * 
 * https://linuxjunkies.wordpress.com/2011/11/21/a-hdfsclient-for-hadoop-using-the-native-java-api-a-tutorial/
 * 
 * Design Ideas
 * 
 * 	-	Need a Canova Conf entry:
 * 		-	hadoop.configuration.path
 * 			-	example: hadoop.configuration.path=/home/hadoop/hadoop/conf/
 * 
 * 
 * @author josh
 *
 */
public class ConfigurationUtil {
	
	public static Configuration generateConfig(String baseConfPath) {
		
		String baseConfPathTrimmed = baseConfPath.trim();
		
		if (false == "/".equals(baseConfPathTrimmed.endsWith("/")) ) {

			baseConfPathTrimmed += "/";
			
		}
		
		Configuration conf = new Configuration();
		conf.addResource(new Path( baseConfPathTrimmed + "core-site.xml"));
		conf.addResource(new Path( baseConfPathTrimmed + "hdfs-site.xml"));
		conf.addResource(new Path( baseConfPathTrimmed + "mapred-site.xml"));
		
		return conf;
		
	}

}
