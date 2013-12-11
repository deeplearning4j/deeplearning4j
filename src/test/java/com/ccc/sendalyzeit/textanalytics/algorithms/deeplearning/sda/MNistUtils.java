package com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.jblas.DoubleMatrix;
import org.springframework.core.io.ClassPathResource;

import com.ccc.sendalyzeit.deeplearning.berkeley.Pair;


public class MNistUtils {

	public static Pair<DoubleMatrix,DoubleMatrix> loadData() throws IOException {
		ClassPathResource resource = new ClassPathResource("/train.csv");
		List<String> lines = IOUtils.readLines(resource.getInputStream());
		int columns = lines.get(0).split(",").length - 1;
		Collections.shuffle(lines);
		Collections.rotate(lines, 3);
		
		DoubleMatrix ret = DoubleMatrix.ones(lines.size(), columns);
		List<String> outcomeTypes = new ArrayList<String>();
	    double[][] outcomes = new double[lines.size()][columns];
		for(int i = 0; i < lines.size(); i++) {
			String line = lines.get(i);
			String[] split = line.split(",");
			
			addRow(ret,i,split);
			
			String outcome = split[0];
			if(!outcomeTypes.contains(outcome))
				outcomeTypes.add(outcome);
			double[] rowOutcome = new double[785];
			rowOutcome[outcomeTypes.indexOf(outcome)] = 1;
			outcomes[i] = rowOutcome;
		}
		return new Pair<>(ret,new DoubleMatrix(outcomes));
	}
	
	
	
	private static void addRow(DoubleMatrix ret,int row,String[] line) {
		double[] vector = new double[4];
		for(int i = 0; i < 4; i++) 
			vector[i] = Double.parseDouble(line[i]);
		
		ret.putRow(row,new DoubleMatrix(vector));
	}
}
