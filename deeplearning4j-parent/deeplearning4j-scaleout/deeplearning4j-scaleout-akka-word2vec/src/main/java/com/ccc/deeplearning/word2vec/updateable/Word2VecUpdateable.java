package com.ccc.deeplearning.word2vec.updateable;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;

import com.ccc.deeplearning.scaleout.iterativereduce.Updateable;
import com.ccc.deeplearning.word2vec.nn.multilayer.Word2VecMultiLayerNetwork;

public class Word2VecUpdateable implements Updateable<Word2VecMultiLayerNetwork> {


	private static final long serialVersionUID = 6547025785641217642L;
	private Word2VecMultiLayerNetwork wrapped;
	private Class<? extends Word2VecMultiLayerNetwork> clazz;


	public Word2VecUpdateable(Word2VecMultiLayerNetwork matrix) {
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
		wrapped = new Word2VecMultiLayerNetwork.Builder()
				.withClazz(clazz).buildEmpty();
		DataInputStream dis = new DataInputStream(new BufferedInputStream(new ByteArrayInputStream(b.array())));
		wrapped.load(dis);
	}

	@Override
	public void fromString(String s) {

	}

	@Override
	public Word2VecMultiLayerNetwork get() {
		return wrapped;
	}

	@Override
	public void set(Word2VecMultiLayerNetwork type) {
		this.wrapped = type;
	}

	@Override
	public void write(DataOutputStream dos) {
		wrapped.write(dos);
	}




}