package org.deeplearning4j.iterativereduce.irunit;

import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Map.Entry;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.InputSplit;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.TextInputFormat;
import org.deeplearning4j.iterativereduce.runtime.ComputableMaster;
import org.deeplearning4j.iterativereduce.runtime.ComputableWorker;
import org.deeplearning4j.iterativereduce.runtime.Updateable;
import org.deeplearning4j.iterativereduce.runtime.io.TextRecordParser;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class IRUnitDriver<T> {

	private static JobConf defaultConf = new JobConf();
	private static FileSystem localFs = null;
	private static Logger log = LoggerFactory.getLogger(IRUnitDriver.class);
	static {
		try {
			defaultConf.set("fs.defaultFS", "file:///");
			localFs = FileSystem.getLocal(defaultConf);
		} catch (IOException e) {
			throw new RuntimeException("init failure", e);
		}
	}

	private static Path workDir = new Path("/tmp/");

	Properties props;

	private ComputableMaster master;
	private ArrayList<ComputableWorker> workers;
	private String app_properties_file = "";
	ArrayList<Updateable> worker_results = new ArrayList<>();
	Updateable master_result = null;
	boolean bContinuePass = true;

	InputSplit[] splits;

	/**
	 * need to load the app.properties file
	 * 
	 * @return
	 */
	public Configuration generateDebugConfigurationObject() {

		Configuration c = new Configuration();

/*		String[] props_to_copy = {
				"app.iteration.count",
				"com.cloudera.knittingboar.setup.FeatureVectorSize",
				"com.cloudera.knittingboar.setup.RecordFactoryClassname",
				"com.cloudera.knittingboar.setup.LearningRate"
		};
	*/	
		 for(Entry<Object, Object> e : props.entrySet()) {
	           // log.info(e.getKey());
			 c.set(e.getKey().toString(), e.getValue().toString());
	        }		
/*
		for ( int x = 0; x < props_to_copy.length; x++ ) {
			if (null != this.props.getProperty(props_to_copy[x])) {
				c.set(props_to_copy[x], this.props.getProperty(props_to_copy[x]));
			} else {
			//	log.info("> Conf: Did not find in properties file - " + props_to_copy[x]);
			}
		}
*/
		
		return c;

	}

	/**
	 * generate splits for this run
	 * 
	 * @param input_path
	 * @param job
	 * @return
	 */
	private InputSplit[] generateDebugSplits(Path input_path, JobConf job) {

		long block_size = localFs.getDefaultBlockSize();

		log.info("default block size: " + (block_size / 1024 / 1024)
				+ "MB");

		// ---- set where we'll read the input files from -------------
		FileInputFormat.setInputPaths(job, input_path);

		// try splitting the file in a variety of sizes
		TextInputFormat format = new TextInputFormat();
		format.configure(job);

		int numSplits = 1;

		InputSplit[] splits = null;

		try {
			splits = format.getSplits(job, numSplits);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return splits;

	}

	public IRUnitDriver(String app_prop) {

		this.app_properties_file = app_prop;
		
	}

	/**
	 * Setup components of the IR app run 1. load app.properties 2. msg arrays
	 * 3. calc local splits 4. setup master 5. setup workers based on number of
	 * splits
	 * 
	 */
	public void Setup() {

		// ----- load the app.properties file

		// String configFile = (args.length < 1) ?
		// ConfigFields.DEFAULT_CONFIG_FILE : args[0];
		this.props = new Properties();
		// Configuration conf = getConf();

		try {
			FileInputStream fis = new FileInputStream(this.app_properties_file);
			props.load(fis);
			fis.close();
		} catch (Exception ex) {
			// throw ex; // TODO: be nice
			log.error("Error loading properties ",ex);
		}

		// setup msg arrays

		// calc splits

		// ---- this all needs to be done in
		JobConf job = new JobConf(defaultConf);
		
		// app.input.path
		
		Path splitPath = new Path( props.getProperty("app.input.path") );

		log.info( "app.input.path = " + splitPath );
		
		// TODO: work on this, splits are generating for everything in dir
		InputSplit[] splits = generateDebugSplits(splitPath, job);

		log.info("split count: " + splits.length);

		try {
			// this.master = (ComputableMaster)
			// custom_master_class.newInstance();

			Class<?> master_clazz = Class.forName(props
					.getProperty("yarn.master.main"));
			Constructor<?> master_ctor = master_clazz
					.getConstructor();
			this.master = (ComputableMaster) master_ctor.newInstance(); // new
																		// Object[]
																		// {
																		// ctorArgument
																		// });
			log.info("Using master class: " + props
					.getProperty("yarn.master.main"));

		} catch (Exception e) {
			e.printStackTrace();
		}

		this.master.setup(this.generateDebugConfigurationObject());

		this.workers = new ArrayList<>();

		log.info("Using worker class: " + props
				.getProperty("yarn.worker.main"));
		
		for (int x = 0; x < splits.length; x++) {
			
			log.info( "IRUnit > Split > " + splits[x].toString() );

			ComputableWorker worker = null;
			Class<?> worker_clazz;
			try {
				worker_clazz = Class.forName(props
						.getProperty("yarn.worker.main"));
				Constructor<?> worker_ctor = worker_clazz
						.getConstructor();
				worker = (ComputableWorker) worker_ctor.newInstance(); // new
																		// Object[]
																		// {
																		// ctorArgument
																		// });

			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			// simulates the conf stuff
			worker.setup(this.generateDebugConfigurationObject());

			// InputRecordsSplit custom_reader_0 = new InputRecordsSplit(job,
			// splits[x]);
			TextRecordParser txt_reader = new TextRecordParser();

			long len = Integer.parseInt(splits[x].toString().split(":")[2]
					.split("\\+")[1]);

			txt_reader.setFile(splits[x].toString().split(":")[1], 0, len);

			worker.setRecordParser(txt_reader);

			workers.add(worker);

			log.info("> Setup Worker " + x);
		} // for

	}

	public void simulateRun() {

		List<Updateable> master_results = new ArrayList<>();
		List<Updateable> worker_results = new ArrayList<>();

		long ts_start = System.currentTimeMillis();

		//log.info("start-ms:" + ts_start);

		int iterations = Integer.parseInt(props
				.getProperty("app.iteration.count"));

		log.info("Starting Epochs (" + iterations + ")...");
		
		for (int x = 0; x < iterations; x++) {

			for (int worker_id = 0; worker_id < workers.size(); worker_id++) {

				Updateable result = workers.get(worker_id).compute();
				worker_results.add(result);


			}

			Updateable master_result = this.master.compute(worker_results,
					master_results);
			

			// process global updates
			for (int worker_id = 0; worker_id < workers.size(); worker_id++) {

				workers.get(worker_id).update(master_result);
				//workers.get(worker_id).IncrementIteration();

			}
/*			
		      if (master_result.get().IterationComplete == 1) {
		          
		          log.info( " -------- end of pass ------- " );

		        // simulates framework checking this and iterating
		          for ( int worker_id = 0; worker_id < workers.size(); worker_id++ ) {
		            
		            bContinuePass = workers.get(worker_id).IncrementIteration();

		          } // for
			*/

		} // for
		
		log.info("Complete " + iterations + " Iterations Per Worker.");

		//String output_path = this.props.getProperty("app.output.path");
		
		//log.info("Writing the output to: " + output_path);
		
		
		// make sure we have somewhere to write the model
		if (null != this.props.getProperty("app.output.path")) {
			
			String output_path = this.props.getProperty("app.output.path");
			
			log.info("Writing the output to: " + output_path);
			

			try {

				Path out = new Path(output_path); 
				FileSystem fs =
						  out.getFileSystem(defaultConf); 
				
				FSDataOutputStream fos;

				fos = fs.create(out);
				  //LOG.info("Writing master results to " + out.toString());
				  master.complete(fos);
				  
				  fos.flush(); 
				  fos.close();


				
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			  

		} else {
			
			log.info("Not Firing Master::Complete() function due to no output path in conf");
			
		}


	}
	
	public ComputableMaster getMaster() {
		
		return this.master;
		
	}
	
	public ArrayList<ComputableWorker> getWorker() {
		
		return this.workers;
		
	}

}