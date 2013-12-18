package com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.jblas;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.nio.ByteBuffer;

import com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.nn.matrix.jblas.BaseMultiLayerNetwork;
import com.ccc.sendalyzeit.textanalytics.ml.scaleout.iterativereduce.Updateable;

public class UpdateableMatrix implements Updateable<BaseMultiLayerNetwork> {

	private BaseMultiLayerNetwork wrapped;
	private int[] rows;
	private Class<? extends BaseMultiLayerNetwork> clazz;
	
	public UpdateableMatrix(BaseMultiLayerNetwork wrapped,int[] rows) {
		this.wrapped = wrapped;
		this.rows = rows;
	}

	public UpdateableMatrix(BaseMultiLayerNetwork matrix) {
		wrapped = matrix;
		if(clazz == null)
			clazz = matrix.getClass();
	}
	
	@Override
	public ByteBuffer toBytes() {
		ByteArrayOutputStream os = new ByteArrayOutputStream();
		DataOutputStream dos = new DataOutputStream(os);
		return ByteBuffer.wrap(os.toByteArray());
		
	}

	@Override
	public void fromBytes(ByteBuffer b) {
		wrapped = new BaseMultiLayerNetwork.Builder<>()
				.withClazz(clazz).buildEmpty();
		DataInputStream dis = new DataInputStream(new BufferedInputStream(new ByteArrayInputStream(b.array())));
		wrapped.load(dis);
	}

	@Override
	public void fromString(String s) {
		
	}

	@Override
	public BaseMultiLayerNetwork get() {
		return wrapped;
	}

	@Override
	public void set(BaseMultiLayerNetwork type) {
		this.wrapped = type;
	}




}
