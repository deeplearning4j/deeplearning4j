Deep Learning for Java
=====================================

DL4J leverages matrix operations built on top of the BLAS linear algebra libraries for faster performance over your standard Java libraries.

Includes the following algorithms:

RBM - Restricted Boltzmann Machines
CRBM - Continuous Restricted Boltzmann Machines
DBN - Deep belief networks; restricted boltzmann machines stacked as layers
CDBN - Continuous Deep Belief Networks; continuous layer at the front
DenoisingAutoEncoders
SDA- Stacked Denoising AutoEncoders
Deep Autoencoders

Typically building a network will look something like this.

        BaseMultiLayerNetwork matrix = new BaseMultiLayerNetwork.Builder<>()
                                .numberOfInputs(conf.getInt(N_IN)).numberOfOutPuts(conf.getInt(OUT)).withClazz(conf.getClazz(CLASS))
                                .hiddenLayerSizes(conf.getIntsWithSeparator(LAYER_SIZES, ",")).withRng(rng)
                                .build();


Configuration is based on the constants specified in DeepLearningConfigurable.

  Usage for the command line app: com.ccc.sendalyzeit.textanalytics.algorithms.deeplearning.sda.matrix.jblas.iterativereduce.actor.ActorNetworkRunnerApp
 
   Options:
       
        Required:
        
        -a algorithm to use: sda (stacked denoising autoencoders),dbn (deep belief networks),cdbn (continuous deep belief networks)
        -i number of inputs (columns in the input matrix)
        -o number of outputs for the network
        -data dataset to train on: options: mnist,text (text files with <label>text</label>, image (images where the parent directory is the label)
        
        Optional:
        
         -fte number of fine tune epochs to train on (default: 100)
        
         -pte number of epochs for pretraining (default: 100)
        
         -r   seed value for the random number generator (default: 123)
        
         -ftl the starter fine tune learning rate (default: 0.1)
        
         -ptl  the starter fine tune learning rate (default: 0.1)
        
         -sp   number of inputs to split by default: 10
        
         -e   number of examples to train on: if unspecified will just train on everything found
        
         DBN/CDBN:
        
         -k the k for rbms (default: 1)
         
         SDA:
        
         -c corruption level (for denoising autoencoders) (default: 0.3)
         
         Cluster:
         
             -h the host to connect to as a master (default: 127.0.0.1)
             -t type of worker
             -ad address of master worker

Training tips:

For classification: Due to heavy sampling leveraged in deep learning, try to have training batches (each row in a training matrix is an individual part of a batch). It is reccomended to have one class of each label in each training batch to minimize sampling error (See Hinton's Practical Guide to RBMs).

Maven central and other support coming soon.

Apache 2.0 licensed. 
