package org.deeplearning4j.tools;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

//import cz.vhr.cbase.Tools;

/**
 * shows possibility how show method and data content
 *
 * 
 *
 * @author clavvis
 */

public class ShowDataExample {
	//
	private static SIS sis;
	//
	private static String moduleCode = "SDE";
	//
	
	public static void main( String[] args ) throws Exception {
		//
		int mtLv = 0;
		//
		String oinfo = "";
		//
		sis = new SIS();
		//
	//	sis.initValues( mtLv, moduleCode, System.out, System.err ); //without file saving
		sis.initValues( mtLv, moduleCode, System.out, System.err, "C:\\Info_Files", moduleCode, "ABC", true, true );
		// System.getProperty( "user.dir" )
		//
		String methodName = moduleCode + "." + "main";
		//
		oinfo = "";
		oinfo += BTools.getMtLvESS( mtLv );
		oinfo += methodName + ": ";
		sis.info( oinfo );
		//
        int minibatchSize = 128;
        int rngSeed       = 12345;
		//
        //MNIST data for training
        DataSetIterator train_DSI = new MnistDataSetIterator( minibatchSize, true, rngSeed );
        //
		DataSet firstTr_DS = train_DSI.next();
		//
		DL4JTools dl4jt = new DL4JTools( sis, moduleCode );
		//
		dl4jt.showDataSet( mtLv, "firstTr_DS", firstTr_DS, 2, 2, 50, 200 );
		//
		INDArray input_ND = firstTr_DS.getFeatures();
		//
		dl4jt.showINDArray( mtLv, "input_ND", input_ND, 1, 50, 800 );
		//
		sis.onStop( mtLv );
		//
	}
}