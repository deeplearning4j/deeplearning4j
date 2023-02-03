package org.eclipse.deeplearning4j.testinit

import org.datavec.image.loader.LFWLoader
import org.deeplearning4j.datasets.fetchers.SvhnDataFetcher
import org.deeplearning4j.datasets.iterator.impl.*
import org.eclipse.deeplearning4j.resources.utils.EMnistSet

class TestDownloadBoostrap {

    fun main(args: Array<String>) {
        var iter = LFWDataSetIterator(1,LFWLoader.NUM_IMAGES)
        var images = TinyImageNetDataSetIterator(10)
        var emnist = EmnistDataSetIterator(EMnistSet.COMPLETE,0,true)
        var mnist = MnistDataSetIterator(1,1)
        var svhn = SvhnDataFetcher()
        var uci = UciSequenceDataSetIterator(10)
        var cifar = Cifar10DataSetIterator(10)


    }

}