package com.ccc.deeplearning.scaleout.iterativereduce.single;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;

import com.ccc.deeplearning.nn.matrix.jblas.BaseNeuralNetwork;
import com.ccc.deeplearning.scaleout.iterativereduce.Updateable;

public class UpdateableSingleImpl implements Updateable<BaseNeuralNetwork> {

	
	private static final long serialVersionUID = 6547025785641217642L;
	private BaseNeuralNetwork wrapped;
	private Class<? extends BaseNeuralNetwork> clazz;
	

	public UpdateableSingleImpl(BaseNeuralNetwork matrix) {
		wrapped = matrix;
		if(clazz == null)
			clazz = matrix.getClass();
	}
	
	@Override
	public ByteBuffer toBytes() {
		ByteArrayOutputStream os = new ByteArrayOutputStream();
		try {
			ObjectOutputStream os2 = new ObjectOutputStream(os);
			os2.writeObject(wrapped);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		
		
		return ByteBuffer.wrap(os.toByteArray());
		
	}

	@Override
	public void fromBytes(ByteBuffer b) {
		wrapped = new BaseNeuralNetwork.Builder<>()
				.withClazz(clazz).buildEmpty();
		DataInputStream dis = new DataInputStream(new BufferedInputStream(new ByteArrayInputStream(b.array())));
		wrapped.load(dis);
	}

	@Override
	public void fromString(String s) {
		
	}

	@Override
	public BaseNeuralNetwork get() {
		return wrapped;
	}

	@Override
	public void set(BaseNeuralNetwork type) {
		this.wrapped = type;
	}

	@Override
	public void write(DataOutputStream dos) {
		wrapped.write(dos);
	}




}
