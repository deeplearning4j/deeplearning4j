package org.datavec.cli.transforms;

import java.util.Collection;

import org.datavec.api.writable.Writable;

public interface Transform {
	
	public void collectStatistics( Collection<Writable> vector );
	
	public void evaluateStatistics();
	
	public void transform( Collection<Writable> vector );

}
