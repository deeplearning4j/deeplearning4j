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

package org.deeplearning4j.hadoop.datasetiterator;

import java.io.FileNotFoundException;
import java.io.InputStream;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;


import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;

/**
 * Baseline support for a dataset iterator iterating over
 * 
 * hdfs data
 * 
 * 
 * @author Adam Gibson
 *
 */
public abstract class BaseHdfsDataSetIterator implements DataSetIterator {

	/**
	 * 
	 */
	private static final long serialVersionUID = 2299460082862304030L;

	private String hdfsUriRootDir;
	private Configuration conf;

	
	/**
	 * Constructs a data applyTransformToDestination iterator with the hdfs uri root directory.
	 * This assumes that individual files are what are being read from wrt
	 * helper methods present.
	 * 
	 * hasNext() and associated data specific methods can be configured
	 * for the specific implementation.
	 * @param hdfsUriRootDir
	 */
	public BaseHdfsDataSetIterator(String hdfsUriRootDir) {
		this.hdfsUriRootDir = hdfsUriRootDir;
	}



	/**
	 * Reads a file from hdfs in to a string.
	 * Note that this is not smart on large files, 
	 * however I will not hand hold you here.
	 * 
	 * 
	 * @param path the path to read from
	 * @return the contents of the file
	 * @throws Exception
	 */
	public String readStringFromPath(String path) throws Exception {
		return readStringFromPath(new Path(path));
	}

	
	
	
	/**
	 * Reads a file from hdfs in to a string.
	 * Note that this is not smart on large files, 
	 * however I will not hand hold you here.
	 * 
	 * 
	 * @param path the path to read from
	 * @return the contents of the file
	 * @throws Exception
	 */
	public String readStringFromPath(Path path) throws Exception {
		InputStream is = openInputStream(path);
		StringWriter writer = new StringWriter();
		IOUtils.copy(is, writer, "UTF-8");
		String theString = writer.toString();
		is.close();
		return theString;
	}

	/**
	 * Opens an input stream for the given path
	 * @param path the path to open an input stream for
	 * @return the opened input stream
	 * @throws Exception
	 */
	public InputStream openInputStream(String path) throws Exception {
		return openInputStream(new Path(path));
	}

	/**
	 * Forget if need to close file system here.
	 * @param path the path to open
	 * @return the input stream for the path
	 * @throws Exception if one occurs
	 */
	public InputStream openInputStream(Path path) throws Exception {
		FileSystem fs = FileSystem.get(conf);
		if(!fs.exists(path))
			throw new FileNotFoundException("File does not exist");
		if(fs.isDirectory(path))
			throw new IllegalArgumentException("Not a file");
		
		InputStream is = fs.open(path);
		return is;

	}

	/**
	 * List all of the files in the 
	 * hdfsUriRootDir directory
	 * @return the list of paths in the directory
	 * @throws Exception if one occurs
	 */
	public List<Path> filesInDir() throws Exception {
		FileSystem fs = FileSystem.get(conf);
		List<Path> paths = new ArrayList<Path>();
		RemoteIterator<LocatedFileStatus> iter = fs.listFiles(new Path(hdfsUriRootDir), true);
		while(iter.hasNext()) {
			LocatedFileStatus l = iter.next();
			paths.add(l.getPath());
		}

		fs.close();
		return paths;

	}




}
