---
title: 基于Clojure的深度学习
layout: cn-default
---

# 基于Clojure的深度学习

Deeplearning4j已通过[DL4CLJ项目](https://github.com/engagor/dl4clj)移植至Clojure。[当前的Github示例](https://github.com/engagor/dl4clj/tree/master/src/dl4clj/examples)介绍了如何配置基于Deeplearning4j的循环神经网络、Word2Vec和ParaVec。关于Clojar的更多详情[参见此处](https://clojars.org/dl4clj)。

## 用Clojure编写的循环神经网络

以下是用Clojure编写的基于DL4CLJ的循环神经网络配置：

      ;; 设置网络配置：
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
      ;; 尚未实现：
      ;; net.setListeners(new ScoreIterationListener(1));

      ;; 显示网络中（及每个层）的参数数量
      (dotimes [i (count (get-layers net))]
        (println "Number of parameters in layer "  i  ": "  (model/num-params (get-layer net i))))
      (println "Total number of network parameters: " (reduce + (map model/num-params (get-layers net))))

## Clojure学习资源

Clojure初学者可以先参考下列资源：

* [4Clojure](http://www.4clojure.com/)－Clojure练习
* [The Joy of Clojure（Clojure编程乐趣）](http://www.joyofclojure.com/)
* [Clojure for the Brave and True（真勇士的Clojure教程）](http://www.braveclojure.com/clojure-for-the-brave-and-true/)

## 其他Deeplearning4j语言

Deeplearning4j还提供[Java](https://github.com/deeplearning4j/deeplearning4j)、[Scala](https://github.com/deeplearning4j/ScalNet)和[Python](./keras)的API，其中Python的API为Keras。
