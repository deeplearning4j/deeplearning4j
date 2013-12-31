package com.ccc.deeplearning.nn.matrix.jblas;

import java.io.InputStream;
import java.io.OutputStream;

public interface Persistable {

	void write(OutputStream os);
	
	void load(InputStream is);
}
