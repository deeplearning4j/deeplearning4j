package com.ccc.deeplearning.plot;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.UUID;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import com.ccc.deeplearning.nn.NeuralNetwork;

/**
 * Credit to :
 * http://yosinski.com/media/papers/Yosinski2012VisuallyDebuggingRestrictedBoltzmannMachine.pdf
 * 
 * 
 * for visualizations
 * @author Adam Gibson
 *
 */
public class NeuralNetPlotter {

	private static 	ClassPathResource r = new ClassPathResource("/scripts/plot.py");
	private static Logger log = LoggerFactory.getLogger(NeuralNetPlotter.class);


	static {
		loadIntoTmp();
	}

	
	public void plot(NeuralNetwork network) {
		plotWeights(network);
		plotHbias(network);
	}
	
	public void plotWeights(NeuralNetwork network) {
		try {

			String filePath = System.getProperty("java.io.tmpdir") + File.separator +  UUID.randomUUID().toString();
			File write = new File(filePath);
			BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(write,true));

			for(int i = 0; i < network.getW().rows; i++) {
				DoubleMatrix row = network.getW().getRow(i);
				StringBuffer sb = new StringBuffer();
				for(int j = 0; j < row.length; j++) {
					sb.append(row.get(j));
					if(j < row.length - 1)
						sb.append(",");
				}
				sb.append("\n");
				String line = sb.toString();
				bos.write(line.getBytes());
				bos.flush();
			}

			bos.close();
			Process is = Runtime.getRuntime().exec("python /tmp/plot.py weights " + filePath);
		
			log.info("Rendering weights " + filePath);
			log.error(IOUtils.readLines(is.getErrorStream()).toString());

			write.deleteOnExit();
		}catch(Exception e) {

		}
	}

	public void plotHbias(NeuralNetwork network) {
		try {
			if(network.getInput() == null)
				throw new IllegalStateException("Unable to plot; missing input");;
				
			
			
			String filePath = System.getProperty("java.io.tmpdir") + File.separator +  UUID.randomUUID().toString();
			File write = new File(filePath);
			BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(write,true));
			DoubleMatrix hbiasMean = network.getInput().mmul(network.getW()).addRowVector(network.gethBias());
			for(int i = 0; i < hbiasMean.rows; i++) {
				DoubleMatrix row = hbiasMean.getRow(i);
				StringBuffer sb = new StringBuffer();
				for(int j = 0; j < row.length; j++) {
					sb.append(row.get(j));
					if(j < row.length - 1)
						sb.append(",");
				}
				sb.append("\n");
				String line = sb.toString();
				bos.write(line.getBytes());
				bos.flush();
			}

			bos.close();
			Process is = Runtime.getRuntime().exec("python /tmp/plot.py hbias " + filePath);
			
			Thread.sleep(10000);
			is.destroy();
			
			
			log.info("Rendering hbias " + filePath);
			log.error(IOUtils.readLines(is.getErrorStream()).toString());

			write.deleteOnExit();
		}catch(Exception e) {
			log.warn("Image closed");

		}
	}
	

	private static void loadIntoTmp() {

		File script = new File("/tmp/plot.py");


		try {
			List<String> lines = IOUtils.readLines(r.getInputStream());
			FileUtils.writeLines(script, lines);

		} catch (IOException e) {
			throw new IllegalStateException("Unable to load python file");

		}

	}

}
