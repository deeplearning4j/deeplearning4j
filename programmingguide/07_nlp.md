---
title: DeepLearning4j NLP
layout: default
---

# DeepLearning4j: Natural Language Processing

In this section, we will learn about Eclipse Deeplearning4j's text processing capabilities. DL4J's natural-language processing (NLP) relies on [ClearTK](https://cleartk.github.io/cleartk/), which is a machine-learning and NLP framework for the Apache [Unstructured Information Management Architecture](https://uima.apache.org/), or UIMA. UIMA provides tools for language identification, language-specific segmentation, sentence-boundary detection and entity detection.

First we will dive into some key NLP concepts like sentence iterators, tokenizers, and vocabulary using DL4J.

- [**Key Concepts**](#key) 
- [**Word2Vec**](#word2vec) 
- [**Bag of Words**](#bag) 

## <a name="key">Key Concepts</a>

### Sentence Iterators

A sentence iterator is used to iterate over a corpus, a collection of written texts, in order to create a list of documents, such as Tweets or newspapers. The purpose of a sentence iterator is to divide the corpus into processable bits and feed text piece by piece to neural networks in the form of vectors. They are used in both [Word2Vec](#word2vec) and [Bag of Words](#bag) models. Below is an example of a sentence iterator that assumes each line in the file is a sentence. 

```
SentenceIterator iter = new LineSentenceIterator(new File("your file"));
```

It is also possible to create a sentence iterator that iterates over multiple files (see example below). This iterator will parse all the files in the directory line by line and return sentences from each file. 

```
SentenceIterator iter = new FileSentenceIterator(new File("your directory"));
```

For more complex uses, we recommend the [UimaSentenceIterator](https://deeplearning4j.org/doc/org/deeplearning4j/text/sentenceiterator/UimaSentenceIterator.html), which can be used for tokenization, part-of-speech tagging, and more. The `UimaSentenceIterator` iterates over a set of files and can segment sentences. We show how to create a `UimaSentenceIterator` below.

```
SentenceIterator iter = UimaSentenceIterator.create("path/to/your/text/documents");
```

It can also be instantiated directly.

```
SentenceIterator iter = new UimaSentenceIterator(path,AnalysisEngineFactory.createEngine(AnalysisEngineFactory.createEngineDescription(TokenizerAnnotator.getDescription(), SentenceAnnotator.getDescription())));
```

The behavior of the `UimaSentenceIterator` can be customized using the `AnalysisEngine` that is passed to it. The `AnalysisEngine` is an UIMA abstraction for a text-processing pipeline. DL4J comes with standard analysis engines for all of the comon tasks, which allows you to customize which text is passed in and how sentences are defined. 

### Tokenization

Depending on the application, we may be able skip using sentence iterators altogether and go straight to tokenization. Tokenization breaks down text into individual words. In order to obtain tokens, we require the use of a tokenizer. An example of a tokenizer is shown below.

```
//tokenization with lemmatization,part of speech taggin,sentence segmentation
TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
Tokenizer tokenizer = tokenizerFactory.tokenize("mystring");

//iterate over the tokens
while(tokenizer.hasMoreTokens()) {
  String token = tokenizer.nextToken();
}
      
//get the whole list of tokens
List<String> tokens = tokenizer.getTokens();
```

We can create tokenizers for strings using a `TokenizerFactory` and then obtain tokens by calling on the functions of the tokenizer. We can either iterate over the tokens or get an entire list of tokens at once. With those tokens, it is possible to create the vocabulary of a document. 

### Vocabulary

The mechanism for general-purpose, natural-language tasks in DL4J is the vocabulary or vocab cache. The vocab cache can be used to store tokens, the frequencies of words, document occurrences, and more. As you iterate over tokens of a text, you need to decide whether the tokens should be included in the vocab cache. The usual criterion is if a token occurs more than a predetermined frequency in the corpus, then it is added to the vocab. If the token does not occur frequently enough, then the token will not be a vocab word. In order to track tokens, we can use code as shown below.

```
addToken(new VocabWord(1.0,"myword"));
```

If a certain token comes up enough times, then it is added to the vocab. We can see that adding the word to the index sets the index and is then declared as a vocab word.

```
addWordToIndex(0, Word2Vec.UNK);
putVocabWord(Word2Vec.UNK);
```

## <a name="word2vec">Word2Vec</a>

Now that we've covered some basic NLP concepts using DL4J, we'll dive into Word2Vec. Word2Vec is a neural network with two hidden layers that processes text. Its input is a corpus of text, and its outputs are numerical feature vectors representing words in the corpus. The goal of Word2Vec is to process text into a format deep neural networks can understand. It can be applied to sentences, code, genes, and other symbolic or verbal series.

Word2Vec attempts to group vectors from similar words together while separating vectors from words that are dissimilar. Thus these vectors are simply numerical representations of individual words and are called neural network embeddings of words. Using these vectors, we can try to determine how similar two words are. One metric is [cosine similarity](https://deeplearning4j.org/glossary.html#cosine). A word has a cosine similarity of 1 with itself and lower cosine similarities with words that are dissimilar to it.

Word2Vec creates feature representations by training words against other words in the corpus. It can do this in two ways, either by predicting a target word from a context (neighboring words in a corpus) or predicting the context from a word. DL4J uses the latter method, since it has been shown to produce more accurate results using large datasets. If the context cannot be accurately predicted from the feature vector, the feature vector's components are adjusted so that the feature vectors of the context are closer in value.

A successful application of Word2Vec could map words like `oak`, `elm`, and `birch` in one cluster and other words like `war`, `conflict`, and `strife` in another cluster. Thus, Word2Vec attempts to represent qualitative similarities of words quantitatively. Word2Vec is implemented through the use of sentence iterators, tokenizers, and vocabulary.  Since these concepts have been explained above, let's dive into the Word2Vec code. 

### Word2Vec Code

As always, first we need to load the data. We assume `raw_sentences.txt` is a text file that contains raw sentences.  We will also use a sentence iterator to iterate through the corpus of text.

```
String filePath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();

log.info("Load & Vectorize Sentences....");
SentenceIterator iter = new BasicLineIterator(filePath);
```

Word2Vec requires words as input, so the data needs to be additionally tokenized. The tokenizer below will create a new token every time a white space is found.  

```
// Split on white spaces in the line to get words
TokenizerFactory t = new DefaultTokenizerFactory();
t.setTokenPreProcessor(new CommonPreprocessor());
```
Now we can configure the Word2Vec model.

```
Word2Vec vec = new Word2Vec.Builder()
  .minWordFrequency(5)
  .iterations(1)
  .layerSize(100)
  .windowSize(5)
  .iterate(iter)
  .tokenizerFactory(t)
  .build();
```

There are a lot of parameters to Word2Vec, and we will explain them here: 

`minWordFrequency` is the minimum number of times a word must appear in the corpus. Thus, if a word occurs fewer than 5 times, the feature vector representation of the word will not be learned by Word2Vec. Learning an appropriate feature vector representation requires that words appear in multiple contexts so that useful features can be learned. 

The `iterations` parameter controls the number of times the network will update its coefficients for a batch of the data. If the number of iterations is too few for the data, then the algorithm might not learn the features effectively, but if there are too many iterations, the training time might be too long. 

`layerSize` specifies the number of dimensions of the feature vector of a word. Thus, a `layerSize` of 100 means that a word will be represented by a 100-dimensional vector. 

`windowSize` is the amount of words that are processed at a time. This is similar to the batch size of the data.

We pass the previously intiialized sentence iterator and tokenizer to the algorithm as well. To actually start the training process, we can just call the fit function of Word2Vec as shown below.

```
log.info("Fitting Word2Vec model....");
vec.fit();
```
Next, we need to evaluate the learned feature representation of the words. 

```
WordVectorSerializer.writeWordVectors(vec, "pathToWriteto.txt");

log.info("Closest Words:");
Collection<String> lst = vec.wordsNearest("day", 8);
 System.out.println(lst);
```

The code above outputs the words closest to an input word. In this case, we are studying the word `day` and the output of the code is `[night, week, year, game, season, during, office, until]`. We can then test the cosine similarities of these words to see if the network perceives similar words as actually similar. 

```
double cosSim = vec.similarity("day", "night");
System.out.println(cosSim);
//output: 0.7704452276229858
```

To visualize the data in a 2 or 3 dimensional space, [TSNE](https://lvdmaaten.github.io/tsne/) (t-Distributed Stochastic Neighbor Embedding) can be used. 

```
log.info("Plot TSNE....");
BarnesHutTsne tsne = new BarnesHutTsne.Builder()
  .setMaxIter(1000)
  .stopLyingIteration(250)
  .learningRate(500)
  .useAdaGrad(false)
  .theta(0.5)
 .setMomentum(0.5)
  .normalize(true)
  .usePca(false)
  .build();
vec.lookupTable().plotVocab(tsne);
```

Lastly, we will want to save our model as follows.

```
WordVectorSerializer.writeWord2VecModel(vec, "pathToSaveModel.txt");
```

The vectors will be saved to `pathToSaveModel.txt`. Each word will be on its own line and will be followed by its vector representation. To reload the vectors, use the following line.

```
Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel("pathToSaveModel.txt");
```

## <a name="bag">Bag of Words</a>

[Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) (BoW) is an algorithm that counts how often a word appears in a corpus. Using these frequencies of words, we can compare different documents as a whole and gauge their similarities for applications such as topic modeling and document classification. Thus, BoW is one method to prepare text as input for a neural network.

BoW lists the words and their frequencies for each document. Each vector of frequencies is then normalized before being fed into a neural network.  Therefore, these counts can now be thought of as probabilities that a word appear in a document. The idea is that words with high probabilities will activate nodes in a neural network, thus influencing how a document is ultimately classified. 

### Term Frequency Inverse Document Frequency

Term Frequency Inverse Document Frequency (TF-IDF) is another way to classify a document by topic using the words it contains. However, instead of merely counting frequencies naively, TF-IDF measures the relevance of a word in a document. 

To calculate a TF-IDF measure, the frequencies of the words are computed in another way, giving weight to words that appear in just a few documents and discounting words that appear frequently across all documents. For example, words like "the" and "and" would be heavily discounted. The basic notion is that words with high relevance will end up being words that are frequent and distinctive with respect to an individual document, rather than evenly distributed across a corpus. The scores are then normalized so they add up to one. 

The simple formula for a TF-IDF is shown below:

```
W = tf(log(N/ df))
```
where `tf` stands for the frequency of a word in a document, `N` represents the total number of documents, and `df` is the total number of documents containing the word. These are then used as features which are fed into a neural network.

### BoW Code

The code to set up a BoW looks like what is shown below. 

```
public class BagOfWordsVectorizer extends BaseTextVectorizer {
  public BagOfWordsVectorizer(){}
  protected BagOfWordsVectorizer(VocabCache cache,
    TokenizerFactory tokenizerFactory,
    List<String> stopWords,
    int minWordFrequency,
    DocumentIterator docIter,
    SentenceIterator sentenceIterator,
    List<String> labels,
    InvertedIndex index,
    int batchSize,
    double sample,
    boolean stem,
    boolean cleanup) {
  super(cache, tokenizerFactory, stopWords, minWordFrequency, docIter, sentenceIterator,
  labels,index,batchSize,sample,stem,cleanup);
    }
```

Overall, BoW produces `wordcounts` and can be effectively used to classify documents as a whole. To identify content or subsets of content instead, Word2Vec should be used.

### DL4J's Programming Guide  

[1. Intro: Deep Learning, Defined](01_intro)
[2. Process Overview](02_process)
[3. Program & Code Structure](03_code_structure)
[4. Convolutional Network Example](04_convnet)
[5. LSTM Network Example](05_lstm)
[6. Feed-Forward Network Example](06_feedforwardnet)
[7. Natural Language Processing](07_nlp)
[8. AI Model Deployment](08_deploy)
[9. Troubleshooting Neural Networks](09_troubleshooting)
