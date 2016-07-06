package org.nd4j.etl4j.cli.transforms;

import java.util.Collection;

import org.nd4j.etl4j.api.writable.Writable;

public interface Transform {
	
	public void collectStatistics( Collection<Writable> vector );
	
	public void evaluateStatistics();
	
	public void transform( Collection<Writable> vector );

}
