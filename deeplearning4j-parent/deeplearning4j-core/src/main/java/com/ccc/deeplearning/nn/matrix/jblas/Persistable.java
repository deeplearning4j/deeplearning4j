package com.ccc.deeplearning.nn.matrix.jblas;

import java.io.InputStream;
import java.io.OutputStream;
import java.io.Serializable;

public interface Persistable extends Serializable {

	void write(OutputStream os);
	
	void load(InputStream is);
}
