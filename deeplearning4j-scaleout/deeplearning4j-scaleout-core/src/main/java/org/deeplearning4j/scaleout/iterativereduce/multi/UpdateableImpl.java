package org.deeplearning4j.scaleout.iterativereduce.multi;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;

import org.apache.commons.lang3.SerializationUtils;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.scaleout.iterativereduce.Updateable;

/**
 * Wrapper for message passing on a model
 */
public class UpdateableImpl implements Updateable<BaseMultiLayerNetwork> {


	private static final long serialVersionUID = 6547025785641217642L;
	private BaseMultiLayerNetwork wrapped;
	private Class<? extends BaseMultiLayerNetwork> clazz;


	public UpdateableImpl(BaseMultiLayerNetwork wrapped) {
		if(wrapped == null)
           throw new IllegalArgumentException("Null argument not allowed");
        this.wrapped = wrapped;
		if(clazz == null)
			clazz = wrapped.getClass();
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
		wrapped = new BaseMultiLayerNetwork.Builder<>()
				.withClazz(clazz).buildEmpty();
		DataInputStream dis = new DataInputStream(new BufferedInputStream(new ByteArrayInputStream(b.array())));
		wrapped.load(dis);
	}

	@Override
	public void fromString(String s) {

	}

	@Override
	public  BaseMultiLayerNetwork get() {
		return wrapped;
	}

	@Override
	public void set(BaseMultiLayerNetwork type) {
		this.wrapped = type;
	}

	@Override
	public void write(DataOutputStream dos) {
		wrapped.write(dos);
	}

	@Override
	public UpdateableImpl clone()  {
		return SerializationUtils.clone(this);
	}




}
