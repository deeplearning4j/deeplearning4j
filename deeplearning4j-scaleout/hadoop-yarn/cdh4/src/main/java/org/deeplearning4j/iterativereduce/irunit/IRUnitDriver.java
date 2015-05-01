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

package org.deeplearning4j.iterativereduce.irunit;

import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;
import java.util.Map.Entry;
import java.util.Properties;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.canova.api.records.reader.RecordReader;
import org.deeplearning4j.iterativereduce.impl.reader.CanovaRecordReader;
import org.deeplearning4j.iterativereduce.runtime.ComputableMaster;
import org.deeplearning4j.iterativereduce.runtime.ComputableWorker;
import org.deeplearning4j.scaleout.api.ir.Updateable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * IRUnitDriver. Simulate an iterative reduce engine on hadoop
 *
 * @author Josh Patterson
 * @author Adam Gibson
 */
public class IRUnitDriver {


	public final static String APP_OUTPUT_PATH = "app.output.path";
	public final static String APP_NUM_ITERATIONS = "app.iteration.count";
	public final static String APP_MAIN = "yarn.worker.main";
	public final static String MASTER_MAIN = "yarn.master.main";
	public final static String APP_INPUT_PATH = "app.input.path";
	public final static String APP_RECORD_READER = "app.recordreader.class";

	private static JobConf defaultConf = new JobConf();
	private static FileSystem localFs = null;
	private static final Logger log = LoggerFactory.getLogger(IRUnitDriver.class);


	private Properties props;
	private ComputableMaster master;
	private ArrayList<ComputableWorker> workers;
	private String appPropertiesFile = "";

	static {
		try {
			defaultConf.set("fs.defaultFS", "file:///");
			localFs = FileSystem.getLocal(defaultConf);
		} catch (IOException e) {
			throw new RuntimeException("init failure", e);
		}
	}





	/**
	 * need to load the app.properties file
	 *
	 * @return {@link Configuration}
	 */
	public Configuration getConfiguration() {
		Configuration c = new Configuration();
		for(Entry<Object, Object> e : props.entrySet())
			c.set(e.getKey().toString(), e.getValue().toString());
		return c;

	}

	/**
	 * generate splits for this run
	 *
	 * @param inputPath
	 * @param job
	 * @return array of {@link InputSplit}
	 */
	private InputSplit[] generateDebugSplits(Path inputPath, JobConf job) {

		long block_size = localFs.getDefaultBlockSize(inputPath);

		log.info("default block size: " + (block_size / 1024 / 1024) + "MB");

		// ---- set where we'll read the input files from -------------
		FileInputFormat.setInputPaths(job, inputPath);

		// try splitting the file in a variety of sizes
		TextInputFormat format = new TextInputFormat();
		format.configure(job);

		int numSplits = 1;

		InputSplit[] splits = null;

		try {
			splits = format.getSplits(job, numSplits);
		} catch (IOException e) {
			log.error("Error loading properties ",e);

		}

		return splits;

	}

	public IRUnitDriver(String appProp) {
		this.appPropertiesFile = appProp;

	}

	/**
	 * setup components of the IR app run 1. load app.properties 2. msg arrays
	 * 3. calc local splits 4. setup master 5. setup workers based on number of
	 * splits
	 *
	 */
	public void setup() {


		this.props = new Properties();

		try {
			FileInputStream fis = new FileInputStream(this.appPropertiesFile);
			props.load(fis);
			fis.close();
		} catch (Exception ex) {
			log.error("Error loading properties ",ex);
		}

		for(Object s : props.keySet())
			defaultConf.set(s.toString(),props.getProperty(s.toString()));


		// setup msg arrays

		// calc splits

		// ---- this all needs to be done in
		JobConf job = new JobConf(defaultConf);
		RecordReader recordReader;
		try {
			recordReader = (RecordReader) Class.forName(defaultConf.get(APP_RECORD_READER)).newInstance();
		} catch (Exception e) {
			throw new RuntimeException(e);
		}


		Path splitPath = new Path( props.getProperty(APP_INPUT_PATH) );

		log.info( APP_INPUT_PATH + " = " + splitPath );

		InputSplit[] splits = generateDebugSplits(splitPath, job);

		log.info("split count: " + splits.length);

		try {
			Class<?> master_clazz = Class.forName(props.getProperty(MASTER_MAIN));
			Constructor<?> master_ctor = master_clazz.getConstructor();
			this.master = (ComputableMaster) master_ctor.newInstance();

			log.info("Using master class: " + props.getProperty(MASTER_MAIN));

		} catch (Exception e) {
			log.error("Error loading master",e);
		}

		this.master.setup(getConfiguration());
		this.workers = new ArrayList<>();

		log.info("Using worker class: " + props.getProperty(APP_MAIN));

		for (int x = 0; x < splits.length; x++) {

			log.info( "IRUnit > Split > " + splits[x].toString() );

			ComputableWorker worker = null;
			Class<?> worker_clazz;
			try {
				worker_clazz = Class.forName(props.getProperty(APP_MAIN));
				Constructor<?> workerConstructor = worker_clazz.getConstructor();
				worker = (ComputableWorker) workerConstructor.newInstance();
			} catch (Exception e) {
				log.error("Error loading worker",e);
			}

			// simulates the conf stuff
			worker.setup(getConfiguration());
			CanovaRecordReader reader = new CanovaRecordReader(recordReader);
			try {
				reader.initialize(splits[x]);
			} catch (IOException | InterruptedException e) {
				e.printStackTrace();
			}

			worker.setRecordReader(reader);
			workers.add(worker);

			log.info("> setup Worker " + x);
		}

	}

	/**
	 * Simulates an iterative reduce run
	 */
	public void simulateRun() {

		List<Updateable> master_results = new ArrayList<>();
		List<Updateable> workerResults = new ArrayList<>();

		int iterations = Integer.parseInt(props.getProperty(APP_NUM_ITERATIONS));

		log.info("Starting Epochs (" + iterations + ")...");

		for (int x = 0; x < iterations; x++) {

			for (ComputableWorker worker : workers) {
				Updateable result = worker.compute();
				workerResults.add(result);
			}

			Updateable master_result = this.master.compute(workerResults, master_results);


			// process global updates
			for (ComputableWorker worker : workers) {
				worker.update(master_result);
			}


			log.info("Complete " + iterations + " Iterations Per Worker.");

			// make sure we have somewhere to write the model
			if (null != props.getProperty(APP_OUTPUT_PATH)) {

				String outputPath = props.getProperty(APP_OUTPUT_PATH);

				log.info("Writing the output to: " + outputPath);

				try {
					Path out = new Path(outputPath);
					FileSystem fs = out.getFileSystem(defaultConf);

					FSDataOutputStream fos;

					fos = fs.create(out);
					master.complete(fos);

					fos.flush();
					fos.close();


				} catch (IOException e) {
					log.error("IO Exception loading path",e);
				}
			}

			else
				log.info("Not Firing Master::Complete() function due to no output path in conf");

		}
	}

	public ComputableMaster getMaster() {
		return master;
	}

	public List<ComputableWorker> getWorker() {
		return workers;
	}

}