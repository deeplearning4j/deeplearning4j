package org.deeplearning4j.scaleout.iterativereduce.multi.gradient;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;

import org.deeplearning4j.nn.gradient.NeuralNetworkGradient;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;


public class UpdateableGradientImpl implements Updateable<NeuralNetworkGradient> {


	private static final long serialVersionUID = 6547025785641217642L;
	private NeuralNetworkGradient wrapped;
	private Class<? extends NeuralNetworkGradient> clazz;


	public UpdateableGradientImpl(NeuralNetworkGradient matrix) {
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
		DataInputStream dis = new DataInputStream(new BufferedInputStream(new ByteArrayInputStream(b.array())));
		wrapped.load(dis);
	}

	@Override
	public void fromString(String s) {

	}

	@Override
	public NeuralNetworkGradient get() {
		return wrapped;
	}

	@Override
	public void set(NeuralNetworkGradient type) {
		this.wrapped = type;
	}

	@Override
	public void write(DataOutputStream dos) {
		wrapped.write(dos);
	}




}
