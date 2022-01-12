package org.eclipse.deeplearning4j.testinit

import org.datavec.image.loader.LFWLoader
import org.deeplearning4j.datasets.fetchers.EmnistDataFetcher
import org.deeplearning4j.datasets.fetchers.SvhnDataFetcher
import org.deeplearning4j.datasets.iterator.impl.*

class TestDownloadBoostrap {

    fun main(args: Array<String>) {
        val iter = LFWDataSetIterator(1,LFWLoader.NUM_IMAGES)
        val images = TinyImageNetDataSetIterator(10)
        val emnist = EmnistDataSetIterator(EmnistDataSetIterator.Set.COMPLETE,0,true)
        val mnist = MnistDataSetIterator(1,1)
        val svhn = SvhnDataFetcher()
        val uci = UciSequenceDataSetIterator(10)
        val cifar = Cifar10DataSetIterator(10)


    }

}