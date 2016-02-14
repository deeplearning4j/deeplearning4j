---
title: The WellDressed Recommendation Engine
layout: default
---

# The WellDressed Recommendation Engine

[*By Stephan Duquesnoy*](https://twitter.com/stephanduq)

## Background

I have a background in theater and IT. 

For a few years, I ran an art-outsourcing studio aimed at the entertainment industry that created concept art and visual assets for games and movies. I specialized in designing characters. I’ve always been a little bit of a pattern-focused kind of person, so my design work tended to be grounded in design harmony patterns that I studied and described into algorithmic and teachable concepts. 

By way of experiment, I translated the algorithm to a program that could suggest color harmonies in sync with the weather. I added clothes to that program, which created Well Dressed. 

[*Well Dressed*](http://welldressed-app.com/) is an app that suggests clothes to wear or buy based on how you look, as well as on the weather, occasion and your budget. It [debuted at Web Summit 2015 in Ireland](http://goos3d.ie/best-startups-at-web-summit-2015/), and it’s a one-man operation (for now!). As a side-note, even though I’m good with pattern-based thinking, I do not have an academic background. I lack patience and feel the need to create, rather than to completely understand what I’m doing.

## Problem

To build Well Dressed, I needed an many categories for different garment types, as I’m dealing with style advice. 

However, data feeds from merchants offer few categorization options, and on top of that, every data feed is very different from the other. So I needed a solution that would let me analyze the information I could get with the data feeds, and somehow fit it into my database. 

My first solution used keywords with a complex rule-and weight-system. It kind of worked, but it didn’t work well enough. Accuracy was around 65%, and every day I had to recheck every new garment to make sure it was correct, costing me many hours I might have spent on marketing or product development.

## Data Pre-processing

The initial data comes from data feeds from stores worldwide. 

I decided to focus on Title, Description and Category/Keyword fields. Title and Description tend to give valuable, specific hints on what a garment actually is. Category works as a broad identification. 

I do not use image data for garment identification, which seems to counter the normal way to approach categorization of clothes. That’s because fashion designers dislike repeating patterns yearly, and they are constantly looking for ways to visually blend different garment types (in hope of starting a trend): Casual shirts that look like hoodies, jogging pants that look like jeans, leather jackets that look like pea coats, etc. 

The copy of the text tends to be the only true identifier of a garment type. However, I do use the images to identify a style and a design using Opencv. 

All my data is already organized by gender and age before Deeplearning4j sees it.

## Example Data

      Title: Navy Pink Floral Silk Tie
      Description: Every T. M.Lewin tie is made from the finest quality 100% pure silk and hand-finished to perfection, with wool interlining. Our classic ties are available in a range of different colors and patterns. Approx. Width at Widest Point: 8. 5cm Approx. Length: 150cm 100% Silk 100% Wool Interlining Dry Clean Only Catalogue Number - 49089 Approx. Width at Widest Point: 8.5cm; Approx. Length: 150cm; 100% Silk; 100% Wool Interlining; Dry Clean Only; Catalogue Number - 49089
      Categories: Woven Silk Ties

My first step is to remove the brand name from the data. Brands occasionally have names that hold garment words, for example: Armani Jeans. Or they can be so present in the dataset, like Levi’s, that the net will only see jeans if they are from Levi’s

Every word is lowercase. I’m not sure if this has an impact, but it feels sensible. I also remove all numbers and punctuation to make sure I’m only dealing with words.

I also remove stop words: both common stop words, and words that are natural to the industry like Mens, fashion, style, clothes, colors, etc.

The data pre-vectorisation looks like this:

      Title: navy pink floral silk tie
      Description: tie made finest quality pure silk hand-finished perfection wool interlining classic ties available range different approx width widest point cm approx length cm silk wool interlining dry clean only catalogue number approx width widest point cm approx length silk wool interlining dry clean only catalogue number
      Categories: woven silk ties

I use [*word2vec*](../word2vec.html) to create vectors. A word needs to show at least 10 times, and I get 40 vectors for each wordset. I then proceed to sum all vectors per title, description and category, resulting in 120 vectors. In theory, this seemed like a bad idea. I was opposed to this approach because I expected the resulting vectors to be too spread out to be of any use. In practice, however, it works very well.

## Data Pipeline

I need to do some unusual things with the data pipeline. The database has a lot of jeans and t-shirts, but very few tuxedoes and cummerbunds – which is to be expected, considering the different market sizes for each item. But to train properly, all the data still needs to be distributed evenly.

There are 84 garment types in the database. My approach is that every 84 entries in the dataset must have one random garment in it that fits the type. Within those 84 entries, I shuffle the order. This will create an evenly distributed dataset, but at the risk of very rare garments like cummerbunds becoming overfitted. It turned out to be a purely theoretical concern, though, because in practice it works great. 

My full dataset contains 84,000 entries. I simply repeat the 84-random approach a thousand times to build the full dataset.

After the set is done, I normalize it with a scalar. One problem here is that the scalar needs to be remade every time, because every training session the dataset is different due to the random factor. 

## Training

I decided to train many small specific nets rather than creating a big net that can classify any data. 

So I train a net for each store. Since every store has its own way of creating their data feed, each store has its own patterns that are very easy to train for. Because the variation in the data is so small, I can create a highly accurate net quite fast. Other than store-oriented nets, I also create a net for each language; its accuracy is not as good as that of the store nets, as to be expected. But the purpose of the language-oriented net is different. 

•	I use the store nets to classify new garment data. After classification, the garment can immediately be published to the app.
•	I use the language net to classify data in new stores. Then I can quickly create a good dataset for the store to use later in the creation of a store-specific net. Since I need to check every garment anyway, I do not mind a lower accuracy. And it speeds up the overall process immensely.

Here's the current setup for store-oriented nets:

        int outputNum = sergeData.labelSize;
        int numSamples = sergeData.labelSize * 10 * 100;
        int batchSize = sergeData.labelSize * 5;
        int iterations = 500;
        int splitTrainNum = (int) (batchSize * .8);
        int seed = 123;
        int listenerFreq = 500;

    MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .batchSize(batchSize)
                .learningRate(1e-6)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .l1(1e-6)
                .regularization(true)
                .l2(1e-4)
                .list(2)
                .layer(0, new DenseLayer.Builder()
                        .nIn(120)
                        .nOut(80)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .updater(Updater.NESTEROVS)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(80)
                        .nOut(outputNum)
                        .weightInit(WeightInit.XAVIER)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);

A few things to point out:

I use a variable label and batch size. Different stores tend to have different categories. But since the difference between the stores is rather small, it works well with one net design.

A minibatch is 0.5% of all the records. The net takes about 200 epochs to train and produce an accurate model.

Previously, I used 9 vectors as input, but that didn’t produce good results. When I scaled it up to 120 vectors, and when the number of nodes were between the number of features and labels, the results improved a lot.

I still have some fine-tuning to do, but the results are quite good at the moment.

**Results**:

* It takes 2 hours to train a net for a store.
* 95-98% accuracy
* 0.93-0.96 F1 score

**Hardware Specs**:

* MacBook pro 2013
* CPU: 2,4 GHz Intel Core i7
* Memory: 8 GB 1600 MHz DDR3

*(Editor's note: Major e-commerce sites that cannot be named report that DL4J-based recommendations have increased their ad coverage by 200%.)*
