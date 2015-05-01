/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.hadoop.util;


import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.URL;
import java.net.URLDecoder;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.Set;

import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.apache.zookeeper.KeeperException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A applyTransformToDestination of utils for basic hdfs operations
 * @author Adam Gibson
 *
 */
public class HdfsUtils {

	private HdfsUtils(){}
	
	private static final Logger log = LoggerFactory.getLogger(HdfsUtils.class);
	private static Map<Configuration,FileSystem> systems = new HashMap<Configuration,FileSystem>();
	public final static String HDFS_HOST = "hdfs.host";


	public static void setRunLocal(Configuration conf) {
		conf.set("fs.default.name","file:///");
		conf.set("mapred.job.tracker","local");
		conf.set("mapred.system.dir","/tmp/mapred/system");
		conf.set("mapred.local.dir","/tmp/mapred");
		conf.set("hadoop.tmp.dir","/tmp");

	}

	public static void setJarFileFor(Configuration conf,Class<?> jarClass) {
		String jar = findJar(jarClass);
		conf.setClassLoader(Thread.currentThread().getContextClassLoader());
		conf.set("mapred.jar",jar);
	}


	public static String findJar(Class<?> my_class) {
		ClassLoader loader = my_class.getClassLoader();
		String class_file = my_class.getName().replaceAll("\\.", "/") + ".class";
		try {
			for(Enumeration<?> itr = loader.getResources(class_file);
					itr.hasMoreElements();) {
				URL url = (URL) itr.nextElement();
				if ("jar".equals(url.getProtocol())) {
					String toReturn = url.getPath();
					if (toReturn.startsWith("file:")) {
						toReturn = toReturn.substring("file:".length());
					}
					//URLDecoder is a misnamed class, since it actually decodes
					// x-www-form-urlencoded MIME type rather than actual
					// URL encoding (which the file path has). Therefore it would
					// decode +s to ' 's which is incorrect (spaces are actually
					// either unencoded or encoded as "%20"). Replace +s first, so
					// that they are kept sacred during the decoding process.
					toReturn = toReturn.replaceAll("\\+", "%2B");
					toReturn = URLDecoder.decode(toReturn, "UTF-8");
					return toReturn.replaceAll("!.*$", "");
				}
			}
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return null;
	}


	/**
	 * Returns the default file system used to reach hdfs
	 * @param conf the configuration to use
	 * @return conf.getFromOrigin(fs.defaultFS)
	 */
	public static String getHost(Configuration conf) {
		if(conf.get("hdfs.host")==null) {
			try {
				HdfsUtils.setHostForConf(conf);
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}
		return conf.get("hdfs.host");
	}

	public static void setHadoopUser(String userName) {
		System.setProperty("HADOOP_USER_NAME", userName);
	}


	/**
	 * Returns the default file system used to reach hdfs
	 * @param conf the configuration to use
	 * @return conf.getFromOrigin(fs.defaultFS)
	 */
	public static String getHdfs(Configuration conf) {
		return conf.get("fs.defaultFS");
	}

	/**
	 * Set the name node and job tracker for a given host
	 * based on an autodiscovered file: host.properties
	 * @param conf the configuration to applyTransformToDestination
	 * @throws IOException
	 */
	public static void setHostForConf(Configuration conf) throws IOException {
		Properties prop = new Properties();
		InputStream in = HdfsUtils.class.getResourceAsStream("/host.properties");
		if(in==null) {
			throw new IOException("No host.properties found");
		}

		prop.load(in);
		in.close();
		String host = prop.getProperty("hdfs.host","localhost");
		boolean reachable = InetAddress.getByName(host).isReachable(1000);
		if(!reachable) {
			log.warn("Host " + host + " was not reachable! Falling back to localhost");
			host = "localhost";
		}
		log.info("Using host  " + host);
		conf.set("hdfs.host",host);
		conf.set("fs.defaultFS", String.format("hdfs://%s:8020",host));
		conf.set("mapred.job.tracker", String.format("hdfs://%s:8021",host));
	}

	public static void cleanup(Configuration conf) throws Exception {
		if(conf.get(HDFS_HOST) !=null) {
			String hdfs = getHost(conf);
			HdfsLock lock = new HdfsLock(hdfs,2181);
			if(lock.isLocked()) {
				if(log.isDebugEnabled())
					log.debug("Returning paths; already found host");
				List<Path> paths = lock.getPaths();
				FileSystem system = FileSystem.get(conf);
				for(Path path : paths)
					system.delete(path,true);
				lock.delete();
			}
			lock.close();
		}
	}

	/**
	 * Adapted from 
	 * http://terrier.org/docs/v3.5/javadoc/org/terrier/utility/io/HadoopUtility.html#saveClassPathToJob%28org.apache.hadoop.mapred.JobConf%29
	 * @param jobConf
	 * @throws IOException
	 */
	public static List<Path> saveClassPathToJob(JobConf jobConf) throws Exception {
		String hdfs = getHost(jobConf);

		HdfsLock lock = new HdfsLock(hdfs);
		String hdfs2 = getHdfs(jobConf);
		if(jobConf.get(HDFS_HOST) !=null) {
			if(lock.isLocked()) {
				List<Path> ret = lock.getPaths();
				StringBuffer files = new StringBuffer();
				StringBuffer classPath = new StringBuffer();
				for(Path path : ret) {
					files.append(hdfs2 + path.toString());
					files.append(",");
					classPath.append(hdfs2 + path.toString());
					classPath.append(":");
					jobConf.addResource(path.toUri().toURL());
				}
				String classPathToSet = classPath.toString().substring(0, classPath.lastIndexOf(":"));
				String filesToSet = files.toString().substring(0,files.lastIndexOf(","));
				log.info("Setting class path " + classPathToSet);
				log.info("Using files " + filesToSet);
				jobConf.set("mapred.cache.files",filesToSet);
				jobConf.set("mapred.job.classpath.files",classPathToSet);
				return ret;
			}
		}
		List<Path> paths = new ArrayList<Path>();
		log.info("Copying classpath to job");

		final String[] jars = findJarFiles(new String[]{
				System.getenv().get("CLASSPATH"),
				System.getProperty("java.class.path"),
				System.getProperty("surefire.test.class.path")
		});


		final FileSystem defFS = FileSystem.get(jobConf);
		int numFilesWritten = 0;
		for (String jarFile : jars) {
			//class path issues
			if(jarFile.contains("hadoop-client")) {
				log.info("Skipping hadoop-client");
				continue;
			}
			else if(jarFile.contains("mapreduce-run")) {
				log.info("Skipping map reduce run");
				continue;
			}

			Path srcJarFilePath = new Path("file:///"+jarFile);
			String filename = srcJarFilePath.getName();
			Path tmpJarFilePath = makeFile(jobConf, filename);			
			log.info("Uploading " + jarFile + " to " + tmpJarFilePath.toString());
			try {
				defFS.copyFromLocalFile(srcJarFilePath, tmpJarFilePath);
				jobConf.addResource(tmpJarFilePath);
				paths.add(tmpJarFilePath);
				numFilesWritten++;
			}catch(Exception e) {
				for(Path path : paths) {
					if(defFS.exists(path))
						defFS.delete(path,true);
				}

				lock.close();
				log.error(String.format("Exception writing to hdfs; rolling back %d jar files ",numFilesWritten),e);
				throw new IOException("Couldn't write jar file " + jarFile);
			}
		}
		try {
			lock.create(paths);
		}catch(KeeperException.SessionExpiredException e) {
			lock = new HdfsLock(hdfs);
			lock.create(paths);

		}
		
		
		lock.close();
		//resolve any differences by removing  clashing names in the files (archives are removed from files)

		
		Set<Path> remove = new HashSet<Path>();
		for(Path path : paths) {
			boolean exists = false;
			try {
				exists = defFS.exists(path);
			}catch(IllegalArgumentException e) {exists = false;}
			if(!exists)
				remove.add(path);
		}
		paths.removeAll(remove);
		return paths;	
	}



	protected static final String[] checkSystemProperties = {"file", "java", "line", "os", "path", "sun", "user"};
	protected static final Random random = new Random();

	public static Path makeTemporaryFile(Configuration jobConf, String filename) throws IOException
	{
		final int randomKey = jobConf.getInt("terrier.tempfile.id", random.nextInt());
		jobConf.setInt("terrier.tempfile.id", randomKey);
		FileSystem defFS = FileSystem.get(jobConf);
		final Path tempFile = new Path("/tmp/"+(randomKey)+"-"+filename);
		defFS.deleteOnExit(tempFile);
		return tempFile;
	}
	public static Path makeFile(Configuration jobConf, String filename) throws IOException
	{
		final int randomKey = jobConf.getInt("terrier.tempfile.id", random.nextInt());
		jobConf.setInt("terrier.tempfile.id", randomKey);
		final Path tempFile = new Path("/tmp/"+(randomKey)+"-"+filename);
		return tempFile;
	}

	protected static String[] findJarFiles(String [] classPathLines)
	{
		Set<String> jars = new HashSet<String>();
		for (String locationsLine : classPathLines)
		{
			if (locationsLine == null)
				continue;
			for (String CPentry : locationsLine.split(":"))
			{
				if (CPentry.endsWith(".jar"))
					jars.add(new File(CPentry).getAbsoluteFile().toString());
			}
		}
		return jars.toArray(new String[0]);
	}
	
	public static void close(Configuration conf) throws Exception {
		FileSystem system = systems.get(conf);
		if(system!=null) {
			system.close();
			systems.remove(conf);

		}
	}

	public static String getHdfsUri(Configuration conf) {
		return conf.get("fs.default.name","127.0.0.1:8020");
	}


	private static FileSystem getFileSystem(Configuration conf) throws Exception{
		FileSystem ret = systems.get(conf);
		if(ret == null) {
			ret = FileSystem.get(conf);
			systems.put(conf,ret);
		}
		return ret;
	}

	public static void ensureUserDirExists(Configuration conf) throws Exception {
		FileSystem fs = getFileSystem(conf);
		if(!fs.exists(new Path(prependUserPath("")))) {
			boolean dirs = fs.mkdirs( new Path(prependUserPath("")));
			if(!dirs)
				throw new IllegalStateException("Couldn't make " + prependUserPath(""));
			FileUtil.chmod(prependUserPath(""), "777");
		}

	}
	
	
	
	
	public static void createFile(String path,Configuration conf) throws Exception  {
		createFile(path,conf,true);

	}
	public static void ensureParentDirectoriesExist(String basePath,Configuration conf) throws Exception {
		File f = new File(basePath);
		if(!pathExists(conf,f.getParent()))
			mkdir(f.getParent(),conf,false);
	}

	public static void createFile(String path,Configuration conf,boolean prepend) throws Exception  {
		FileSystem fs = getFileSystem(conf);
		ensureParentDirectoriesExist(path,conf);
		//returns an output stream, ensure it's closed
		if(!prepend && !pathExists(conf,path)) {
			OutputStream os = fs.create(new Path(path));
			IOUtils.closeQuietly(os);
		}
		else if(!pathExists(conf,prependUserPath( path ) ) ) {
			OutputStream os =  fs.create(new Path( prependUserPath( path ) ) ) ;
			IOUtils.closeQuietly(os);
		}
	}

	public static boolean pathExists(Configuration conf,String path) throws Exception {
		FileSystem fs = getFileSystem(conf);
		boolean ret = fs.exists( new Path(path));
		return ret;
	}
	public static String getUser() {
		return "hdfs";
	}
	public static String getUserPath() {
		return "/user/" + getUser() + "/";
	}
	public static String prependUserPath(String path) {
		return "/user/" + getUser() + "/" + path;
	}

	public static void deleteUserDir(Configuration conf) throws Exception {
		FileSystem fs = getFileSystem(conf);
		if(fs.exists( new Path( prependUserPath( "" ) ) ) ) { 
			boolean delete = fs.delete(new Path( prependUserPath( "" ) ), true);
			if(!delete)
				throw new RuntimeException("Couldn't delete file " + prependUserPath("") );
		}
	}

	public static void writeText(String text,Configuration conf,String file) throws Exception { 
		FileSystem fs = getFileSystem(conf);
		if(pathExists(conf,file)) 
			deletePath(file, conf, false);
		ensureParentDirectoriesExist(file,conf);


		FSDataOutputStream fdos = fs.create(new Path(file), true);
		fdos.writeBytes(text);
		fdos.flush();
		fdos.close();
	}


	public static String getContents(String text,Configuration conf,String file) throws Exception { 
		FileSystem fs = getFileSystem(conf);
		if(!pathExists(conf,file))
			return "";

		InputStream is = fs.open(new Path(file));
		StringBuffer sb = new StringBuffer();
		BufferedReader reader = new BufferedReader( new InputStreamReader(is));
		String line = null;
		while((line = reader.readLine()) != null )
			sb.append(line);
		reader.close();
		is.close();

		return sb.toString();
	}

	public static void deletePath(String path,Configuration conf,boolean prependUserPath) throws Exception {
		FileSystem fs = getFileSystem(conf);
		if(!prependUserPath && pathExists( conf,path ) )
			fs.delete( new Path(path),true);
		else if(pathExists(conf,prependUserPath( path ) ) )
			fs.delete( new Path( prependUserPath( path ) ),true);
	}
	public static void mkdir(String path,Configuration conf,boolean prependUserPath) throws Exception {
		if(prependUserPath) 
			ensureUserDirExists(conf);

		FileSystem fs = getFileSystem(conf);
		if(prependUserPath && !fs.exists(new Path( prependUserPath( path ) ) ) )
			fs.mkdirs(new Path( prependUserPath( path ) ) );
		if(!fs.exists(new Path( path ) ) )
			fs.mkdirs( new Path(path ) );
	}
	public static void deleteDir(String path,Configuration conf) throws Exception {
		deletePath(path,conf,true);
	}
	public static void mkdir(String path,Configuration conf) throws Exception {
		deletePath(path,conf,true);
	}



}
