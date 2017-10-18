---
title: DeepLearning4j NLP
layout: default
---

------

# DeepLearning4j: Natural Language Processing

In this section, we will learn about DL4J's text processing capabilities. DL4J's NLP relies on [ClearTK](https://cleartk.github.io/cleartk/), which is a machine learning and NLP framework for the Apache [Unstructured Information Management Architecture](https://uima.apache.org/) or UIMA. UIMA provides us with the tools to perform language identification, language-specific segmentation, sentence boundary detection and entity detection.

First we will dive into some key DL4J NLP concepts like sentence iterators, tokenizers, and vocab.

- [**Key Concepts**](#key) 
- [**Word2Vec**](#word2vec) 
- [**Bag of Words**](#bag) 

## <a name="key">Key Concepts</a>

### Sentence Iterators

A sentence iterator is used to iterate over a corpus, which is a collection of written texts in order to create a list of documents, such as tweets or newspapers. The purpose of a sentence iterator is to divide text into processable bits. Below is one example of a sentence iterator. This SentenceIterator assumes each line in the file is a sentence. 

```
SentenceIterator iter = new LineSentenceIterator(new File("your file"));
```

It is possible to also create a sentence iterator that iterates over multiple files. An example of this is shown below. This iterator will parse all the files in the directory line by line and return sentences from each. 

```
SentenceIterator iter = new FileSentenceIterator(new File("your directory"));
```

For more complex uses, we recommend the [UimaSentenceIterator](https://deeplearning4j.org/doc/org/deeplearning4j/text/sentenceiterator/UimaSentenceIterator.html), which can be used for tokenization, part-of-speech tagging, and more. The UimaSentenceIterator iterates over a set of files and can segment sentences. We show how to create a UimaSentenceIterator below.

```
SentenceIterator iter = UimaSentenceIterator.create("path/to/your/text/documents");
```

### Tokenization

Depending on the use, we may be able skip the use of sentence iterators altogether and go straight to tokenization. Tokenization is used to break down text into individual words. In order to obtain tokens, we require the use of a tokenizer. An example is shown below.

```
TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
Tokenizer tokenizer = tokenizerFactory.tokenize("mystring");

//iterate over the tokens
while(tokenizer.hasMoreTokens()) {
  String token = tokenizer.nextToken();
}
      
//get the whole list of tokens
List<String> tokens = tokenizer.getTokens();
```

We can create tokenizers for strings using a TokenizerFactory and then obtain tokens by calling on the fnctions of the tokenizer. We can either iterate over the tokens or get an entire list of tokens at once. By obtaining tokens, it is possible to create the vocabulary of a text.

### Vocabulary

The mechanism for general-purpose natural language tasks in DL4J is the vocabulary or vocab cache. The vocab cache can be used to store tokens, frequencies of words, document occurrences, and more. As you iterate over tokens of a text, you need to decide whether the tokens should be included in the vocab cache. The usual criterion is if a token occurs more than a predetermined frequency in the corpus, then it is added in the vocab. If the token does not occur as frequent enough, then the token is not a vocab word. In order to track tokens, we can use code as shown below.

```
addToken(new VocabWord(1.0,"myword"));
```

If a certain token occurs enough times, then it is added to the vocab. As we can see adding the word to the index sets the index and is then declared as a vocab word.

```
addWordToIndex(0, Word2Vec.UNK);
putVocabWord(Word2Vec.UNK);
```

## <a name="word2vec">Word2Vec</a>

Now that we went over some key concept's of DL4J's NLP functionality, we will dive into Word2Vec. Word2Vec is a neural network with 2 hidden layers that processes text. Its input is a corpus of text, and its outputs are numerical feature vectors representing words in the corpus. The goal of Word2Vec is to convert text into a format deep neural networks can understand.

Word2Vec attempts to group vectors from similar words together while separating vectors from words that are dissimilar. Thus these vectors are simply numerical representations of individual words and are called neural network embeddings of words. Using these vectors, we can try to determine how similar two words are. One metric is the [consine similarity](https://deeplearning4j.org/glossary.html#cosine).  A word has a cosine similarity of 1 with itself and a lower cosine similarity with words that are dissimilar.

Word2Vec creates feature representations by training words against other words in the corpus. It can do this in two ways, either by predicting a target word from a context (neighboring words in a corpus) or predicting the context from a word. DL4J uses the latter method, since it has been shown to produce more accurate results using large datasets. If the context cannot be accurately predicted from the feature vector, the feature vector's components are adjusted so that the feature vectors of the context are closer in value.

A successful application of Word2Vec could map words like oak, elm, and birch in one cluster and other words like war, conflict, and strife in another cluster. Thus, Word2Vec attempts to represent qualitative similarities of words quantitatively. Word2Vec is implemented through the use of sentence iterators, tokenizers, and vocabulary.  Since these concepts have been explained above, let's dive into Word2Vec code. 

### Word2Vec Code

As always, first we need to load the data. We assume raw_sentences.txt is a text file that contains raw sentences.  We will also use a sentence iterator to iterate through the corpus of text.

```
String filePath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();

log.info("Load & Vectorize Sentences....");
SentenceIterator iter = new BasicLineIterator(filePath);
```

Word2Vec requires words as input, so the data needs to be additionally tokenized. The below tokenizer will create a new token every time a white space is found.  

```
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

There are a lot of parameters to Word2Vec, and we will take the time to explain them here. minWordFrequency is the minimum number of times a word must appear in the corpus. Thus, if a word occurs fewer than 5 times, the feature vector representation of the word will not be learned by Word2Vec. Learning an appropriate feature vector representation requires words to appear in multiple context so that useful features can be learned. The iterations parameter controls the number of times the network will update its coefficients for a batch of the data. If the number of iterations is too few for the data, then the algorithm might not learn the features effectively but if there are too many iterations the training time might be too long. layerSize specifies the number of dimensions of the feature vector of a word. Thus, the layerSize of 100 means that a word will be represented by a 100 dimensional vector. windowSize is the amount of words that are processed at a time. This is similar in analogous to the batch size of the data.

We can see that we pass the previously intiialized sentence iterator and tokenizer to the algorithm as well. To actually start the training process, we can just call the fit function of Word2Vec as shown below.

```
log.info("Fitting Word2Vec model....");
vec.fit();
```
Next we need to evaluate the learned feature representation of the words. 

```
WordVectorSerializer.writeWordVectors(vec, "pathToWriteto.txt");

log.info("Closest Words:");
Collection<String> lst = vec.wordsNearest("day", 8);
 System.out.println(lst);
```

The above code outputs the closest words to an input word. In this case, we are studying the word "day" and the output of the code is [night, week, year, game, season, during, office, until]. We can then test the cosine similarities of these words to see if the network perceives similar words to be similar. 

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

## <a name="bag">Bag of Words</a>

[Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) (BoW) is an algorithm that counts how often a word appears in a corpus. Using these frequencies of words, we can compare different documents as a whole and gauge their similarities for applications such as topic modeling and document classification. Thus, BoW is one method to prepare text as input for a neural network.

BoW lists the words and their frequencies for each document. Each vector of frequencies is then normalized before being fed into a neural network.  Therefore, these counts can now be thought of a probabilities that a word appears in a document. The idea is that words with high probabilities will activae nodes in a neural network and influence how a document is classified in the end. 

### Term Frequency Inverse Document Frequency

TF-IDF is another way to judge a topic of a document using the words it contains. However, instead of frequency, TF-IDF measures the relevance of a word in a document. To calculate a TF-IDF measure, the frequencies of the words are computed. However, TF-IDF then discounts words that appear frequently across all documents. For example, words like "the" and "and" would be heavily discounted. The basic notion is that words with high relevance will end up being words that are frequent and distinctive with respect to an individual document. The scores are then normalized so they add up to one. 

The simple formula for a TF-IDF is shown below

```
W = tf(log(N/ df))
```
where tf stands for the frequency of a word in a document, N represents the total number of documents, and df is the total number of documents containing the word. These are then used as features which are fed into a neurla network.

###

The code to set up a BoW looks something like what is shown below. 

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
