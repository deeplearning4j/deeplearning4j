---
title: Deep Learning With Clojure
layout: default
---

# Deep Learning With Clojure

Deeplearning4j has been ported to Clojure with the [DL4CLJ project](https://github.com/engagor/dl4clj). The [present Github examples](https://github.com/engagor/dl4clj/tree/master/src/dl4clj/examples) illustrate how to configure recurrent neural networks, word2vec and ParaVec based on Deeplearning4j. More details on the [Clojar are here](https://clojars.org/dl4clj).

## A Recurrent Neural Network in Clojure

Here's what a recurrent neural network configuration looks like in Clojure with DL4CLJ:

      ;; Set up network configuration:
      (def conf (neural-net-configuration
                 {:optimization-algo :stochastic-gradient-descent
                  :iterations 1
                  :learning-rate 0.1
                  :rms-decay 0.95
                  :seed 12345
                  :regularization true
                  :l2 0.001
                  :list 3
                  :layers {0 {:graves-lstm
                              {:n-in (input-columns iter)
                               :n-out lstm-layer-size
                               :updater :rmsprop
                               :activation :tanh
                               :weight-init :distribution
                               :dist {:uniform {:lower -0.08, :upper 0.08}}}}
                           1 {:graves-lstm
                              {:n-in lstm-layer-size
                               :n-out lstm-layer-size
                               :updater :rmsprop
                               :activation :tanh
                               :weight-init :distribution
                               :dist {:uniform {:lower -0.08, :upper 0.08}}}}
                           2 {:rnnoutput
                              {:loss-function :mcxent
                               :activation :softmax
                               :updater :rmsprop
                               :n-in lstm-layer-size
                               :n-out (total-outcomes iter)
                               :weight-init :distribution
                               :dist {:uniform {:lower -0.08, :upper 0.08}}}}}
                  :pretrain false
                  :backprop true}))
      (def net (multi-layer-network conf))
      (init net)
      ;; not yet implemented:
      ;; net.setListeners(new ScoreIterationListener(1));

      ;; Print the  number of parameters in the network (and for each layer)
      (dotimes [i (count (get-layers net))]
        (println "Number of parameters in layer "  i  ": "  (model/num-params (get-layer net i))))
      (println "Total number of network parameters: " (reduce + (map model/num-params (get-layers net))))

## Clojure Resources

Those just beginning with Clojure may want to explore the resources below:

* [4Clojure](http://www.4clojure.com/) - Exercises in Clojure
* [The Joy of Clojure](http://www.joyofclojure.com/)
* [Clojure for the Brave and True](http://www.braveclojure.com/clojure-for-the-brave-and-true/)

## Other Deeplearning4j Languages

Deeplearning4j also offers APIs in [Java](https://github.com/deeplearning4j/deeplearning4j), [Scala](https://github.com/deeplearning4j/ScalNet) and [Python](./keras) with Keras.
