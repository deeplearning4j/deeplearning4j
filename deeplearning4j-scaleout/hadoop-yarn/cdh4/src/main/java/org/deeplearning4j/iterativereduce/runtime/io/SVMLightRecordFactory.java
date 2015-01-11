package org.deeplearning4j.iterativereduce.runtime.io;



import org.nd4j.linalg.api.ndarray.INDArray;


public class SVMLightRecordFactory {

	private boolean useBiasTerm = false;
	private int featureVectorSize = 1;
	//private String schema = "";
	private int inputValues = 0;

	/**
	 * dont need to parse the schema, just trust that the incoming vectors are the right size
	 *
	 */
	public SVMLightRecordFactory(int featureVectorSize) {

		this.featureVectorSize = featureVectorSize;

	}

	public void setUseBiasTerm() {
		this.useBiasTerm = true;
	}

	//	  @Override
	public int getInputVectorSize() {
		return this.inputValues;
	}



	// INDArray vector = Nd4j.create(10);

	/**
	 *
	 * example line: "-1 1:0.43 3:0.12 9284:0.2 # abcdef"
	 *
	 * @param line
	 * @param input_vec
	 * @param output_vec
	 * @throws Exception
	 */
	public void parseFromLine(String line, INDArray input_vec, INDArray output_vec) {

		//String[] inputs_outputs = line.split("\\|");
		// remove comments
		String[] vector_and_comments = line.split("#");

		// now use only the left hand part
		String[] inputs_outputs = vector_and_comments[0].split(" ");

		//System.out.println("in: " + inputs_outputs[0]);
		//System.out.println("out: " + inputs_outputs[1]);

		String label = inputs_outputs[0].trim();
		//String[] outputs = inputs_outputs[1].trim().split(" ");


		// clear it
		input_vec.muli(0.0); // clear vector

		int startFeatureIndex = 0;
		// dont know what to do the the "namespace" "f"
		if (this.useBiasTerm) {
			// input_vec.set(0, 1.0);
			input_vec.putScalar(0, 1.0);
			startFeatureIndex = 1;
		}

		for (int x = 1; x < inputs_outputs.length; x++) {

			//System.out.println("> DEbug > part: " + parts[x]);

			String[] feature = inputs_outputs[x].split(":");

			if ("#".equals( feature[0].trim() ) ) {

				// comment

			} else {
				// get (offset) feature index and hash as neccesary
				int index = (Integer.parseInt(feature[0]) + startFeatureIndex); // % this.featureVectorSize;

				double val = Double.parseDouble(feature[1]);

				if (index < this.featureVectorSize) {

					input_vec.putScalar(index, val);

				} else {

					// Should we throw an exception here?
					System.err.println("Could Hash: " + index + " to " + (index % this.featureVectorSize));

				}

			}

		}


		output_vec.muli(0.0); // clear vector

		double val = 0;
		val = Double.parseDouble( label );
		output_vec.putScalar(0, val);



	}

	public int getFeatureVectorSize() {
		return this.featureVectorSize;
	}

}
