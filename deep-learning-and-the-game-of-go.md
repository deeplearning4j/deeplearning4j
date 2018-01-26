---
title: Building a Go-playing bot with Eclipse Deeplearning4J
layout: default
---

# Building a Go-playing bot with Eclipse Deeplearning4j

As long as there have been computers, programmers have been interested in artificial intelligence (or "AI"): implementing human-like behavior on a computer. Games have long been a popular subject for AI researchers. During the personal computer era, AIs have overtaken humans at checkers, backgammon, chess, and almost all classic board games. But the ancient strategy game Go remained stubbornly out of reach for computers for decades. Then in 2016, Google DeepMind's AlphaGo AI challenged 14-time world champion Lee Sedol and won four out of five games. The next revision of AlphaGo was completely out of reach for human players: it won 60 straight games, taking down just about every notable Go player in the process.

AlphaGo's breakthrough was enhancing classical AI algorithms with machine learning. More specifically, it used modern techniques known as deep learning -- algorithms that can organize raw data into useful layers of abstraction. These techniques are not limited to games at all. You will also find deep learning in applications for identifying images, understanding speech, translating natural languages, and guiding robots.

[Eclipse Deeplearning4J](https://projects.eclipse.org/projects/technology.deeplearning4j) (DL4J) is a powerful, general-purpose deep learning framework for the JVM with which you can build many interesting applications. In this article you will learn how to design a deep learning model for the game of Go that can predict the next move in any given board situation. This model is powered by records of Go games played by professional players. In the end, we will embed this model into a full-blown application that you can play against yourself in your browser!

## Overview

At the end of this article you will know quite a bit about the following topics:

* Understanding the fundamentals of how you can use machine learning, and deep learning in particular, for an interesting problem domain like Go.
* Playing the game of Go at a beginner level.
* Getting a glimpse at what neural networks are and how deep networks can be used for predicting Go moves.
* Understanding what it takes to build and deploy a Go bot.
* Building a deep learning model for Go move prediction with Eclipse DL4J.
* Running a deep learning bot with docker that you can play against.

Many of these topics we can barely scratch the surface of in this article. There's a whole lot more to explore and learn in deep learning and AI. However, many of the techniques that run the strongest Go AIs out there carry over to other applications -- and truly understanding them can be your gateway into more advanced AI topics.

If we've caught your attention and you want to learn more about this fascinating topic, go ahead and check out our new book [Deep Learning and the Game of Go](https://www.manning.com/books/deep-learning-and-the-game-of-go) (Manning).

Readers of this newsletter get a 40% discount off of all formats when using the following code: "smpumperla40". Code for the book and other useful material is freely available in the following [GitHub repository](https://github.com/maxpumperla/deep_learning_and_the_game_of_go). The book itself is written with Python and does not use Eclipse DL4J, but in this article we tackle some aspects covered in detail in the book with Java and DL4J.

![Alt text](./img/dl_go_cover.jpg)

## Machine learning and deep learning

Consider the task of identifying a photo of a friend. This is effortless for most people, even if the photo is badly lit, your friend got a haircut or is wearing a new shirt. But suppose you wanted to program a computer to do the same thing. Where would you even begin? This is the kind of problem that machine learning can solve.

### Traditional programming versus machine learning

Traditionally, computer programming is about applying clear rules to structured data. A human developer programs a computer to execute a set of instructions on data and outputs the desired result. Think of a tax form: every box has a well-defined meaning, and there are detailed rules about how to make various calculations from them. Depending on where you live, these rules may be extremely complicated. It's easy for people to make a mistake here, but this is exactly the kind of task that computer programs excel at.

In contrast to the traditional programming paradigm, machine learning is a family of techniques for inferring a program or algorithm from example data, rather than implementing it directly. So, with machine learning, we still feed our computer data, but instead of imposing instructions and expecting output, we provide the expected output and let the machine find an algorithm by itself.

![Alt text](./img/annotated_traditional_paradigm_sm.png)
*An illustration of the standard programming paradigm that most software developers are familiar with. The developer identifies the algorithm and implements the code; the users supply the data.*

To build a computer program that can identify who's in a photo, we can apply an algorithm that analyzes a large collection of images of your friends and generates a function that matches them. If we do this correctly, the generated function will also match new photos that we've never seen before. Of course, it will have no knowledge of its purpose; all it can do is identify things that are similar to the original images we fed it.

In this situation, we call the images we provide the machine training data and the names of the people on the picture labels. Once we have trained an algorithm for our purpose, we can use it to predict labels on new data to test it. Figure <<figure-ml-schema>> displays this example alongside a schema of the machine learning paradigm.

![Alt text](./img/annotated_ml_paradigm_sm.png)
*An illustration of the machine learning paradigm. During development, we generate an algorithm from a data set and then incorporate that into our final application.*

Machine learning comes in when rules aren't clear; it can solve problems of the "I'll know it when I see it" variety. Instead of programming the function directly, we provide data that indicates what the function should do and then methodically generate a function that matches our data.

In practice, you usually combine machine learning with traditional programming to build a useful application. For our face detection app, we have to instruct the computer on how to find, load, and transform the example images before we can apply a machine learning algorithm. Beyond that, we might use hand-rolled heuristics to separate headshots from photos of sunsets and latte art; then we can apply machine learning to put names to faces. Often a mixture of traditional programming techniques and advanced machine learning algorithms will be superior to either one alone.

### Deep learning

This article is made up of sentences. The sentences are made of words, the words are made of letters, the letters are made up of lines and curve, and ultimately those lines and curves are made up of tiny tinted pixels. When teaching a child to read, we start with the smallest parts and work our way up: first letters, then words, then sentences, and finally complete books. (Normally children learn to recognize lines and curves on their own.) This kind of hierarchy is the natural way for people to learn complex concepts. At each level, we ignore some detail and the concepts become more abstract.

Deep learning applies the same idea to machine learning. Deep learning is a subfield of machine learning that uses a specific family of models: sequences of simple functions chained together. These chains of functions are known as neural networks because they were loosely inspired by the structure of natural brains. The core idea of deep learning is that these sequences of functions can analyze a complex concept as a hierarchy of simpler ones. The first layer of a deep model can learn to take raw data and organize it in basic ways -- like grouping dots into lines. Each successive layer organizes the previous layer into more advanced and abstract concepts.

The amazing thing about deep learning is that you do not need to know what the intermediate concepts are in advance. If you select a model with enough layers, and provide enough training data, the training process will gradually organize the raw data into increasingly high-level concepts. But how does the training algorithm know what concepts to use? It doesn't really; it just organizes the input in any way that helps it match the training examples better. So there is no guarantee this representation matches the way humans would think about the data.

![Alt text](./img/dl_pipeline_sm.jpg)
*Deep learning and representation learning.*

## A lightning introduction to the Game of Go

The rules of Go are famously simple. In short, two players alternate placing black and white stones on a board, starting with the black player. The goal is to surround as much of the board as possible with your own stones. Although the rules are simple, Go strategy has endless depth, and we don't even attempt to cover it here.

### The board

A Go board is a square grid. Stones go on the intersections, not inside the squares. The standard board is 19×19, but sometimes players use a smaller board for a quick game. The most popular smaller options are 9×9 and 13×13 boards.

![Alt text](./img/board_with_stones.png) 
*A standard 19 × 19 Go board. The intersections marked with the dots are the star points: they are solely for players' reference. Stones go on the intersections.*

Placing and capturing stones
One player plays with black stones and the other plays with white stones. The two players alternate placing stones on the board, starting with the black player. Stones don't move once they are on the board, although they can be captured and removed entirely. To capture your opponent's stones, you must completely surround them with your own. Here's how that works.

Stones of the same color that are touching are considered connected together. For the purposes of connection, we only consider straight up, down, left, or right; diagonals don't count. Any empty point touching a connected group is called a liberty of that group. Every group needs at least one liberty to stay on the board. So you can capture your opponent's stones by filling their liberties.

![Alt text](./img/liberties.png)
*The three black stones are connected. They have four liberties on the points marked with squares. White can capture the black stones by placing white stones on all the liberties.*

When you place a stone in the last liberty of an opponent's group, that group is captured and removed from the board. The newly empty points are then available for either player to play on (so long as the move is legal). On the flip side, you may not play a stone that would have zero liberties, unless you are completing a capture.

![Alt text](./img/two_eyes.png)
*The white stones on the left can never be captured: black can play at neither A nor B. A black stone there would have no liberties and is, therefore, an illegal play. On the other hand, black can play at C to capture five white stones.*

There's an interesting consequence of the capturing rules. If a group of stones has two completely separate internal liberties, it can never be captured. See the figure above: black can't play at A, because that black stone would have no liberties. Nor can black play at B. So black has no way to fill the last two liberties of the white group. These internal liberties are called eyes. In contrast, black can play at C to capture five white stones. That white group has only one eye and is doomed to get captured at some point.

Although it's not explicitly part of the rules, the idea that a group with two eyes can't be captured is the most basic part of Go strategy. In fact, this is the only strategy we will specifically code into our bot's logic. All the more advanced Go strategies will be inferred through machine learning.

### Ending the game and counting

Either player may pass any turn instead of placing a stone. When both players pass consecutive turns, the game is over. Before scoring, the players identify any dead stones: stones that have no chance of making two eyes or connecting up to friendly stones. Dead stones are treated exactly the same as captures when scoring the game. If there's a disagreement, the players can resolve it by resuming play. But this is very rare: if the status of any group is unclear, players will usually try to resolve it before passing.

The goal of the game is to surround a larger section of the board than your opponent. There are two ways to add up the score, but they nearly always give the same result.

The most common counting method is territory scoring. In this case, you get one point for every point on the board that is completely surrounded by your own stones, plus one point for every opponent's stone that you captured. The player with more points in the winner.

Here's a full 9x9 game with explanations that illustrate all these concepts in a concrete example. Use the arrows below the Go board to navigate through the moves.
  <div style="text-align: center; width: 100%">
  <div style="display: inline-block;">
    <div style="float: left;">
      <div id="board">
      </div>

      <div class="commentary" style="height:80px; width:570px;">
      <div id="lastMove">
      </div>
      <div id="comments">
      </div>
      </div>

      <p class="controls">
      <a href="#" onclick="move(-5); return false;"><i class="fa fa-backward"></i></a>
      <a href="#" onclick="move(-1); return false;"><i class="fa fa-step-backward"></i></a>
      <strong id="move">1</strong> / <strong id="moves">1</strong>
      <a href="#" onclick="move(1); return false;"><i class="fa fa-step-forward"></i></a>
      <a href="#" onclick="move(5); return false;"><i class="fa fa-forward"></i></a>
      </p>
    </div>

  </div>
  </div>

  <script type="text/javascript" src="/community/eclipse_newsletter/2018/january/dist/jgoboard-latest.js"></script>
  <script type="text/javascript" src="/community/eclipse_newsletter/2018/january/large/board.js"></script>
  <script type="text/javascript" src="/community/eclipse_newsletter/2018/january/medium/board.js"></script>
  <script type="text/javascript">

  var gameRecord = [
  ["G7"],
  ["C7"],
  ["C3"],
  ["G3", "The players start in opposite corners, a typical opening."],
  ["H4", "Black attempts to make some territory on the right side."],
  ["G4", "White responds by strengthening the G4 stone."],
  ["H3"],
  ["H2"],
  ["G5"],
  ["D4", "This move has a loose connection to the G4 stones, and also puts pressure on the black C3 stone."],
  ["D3"],
  ["E2", "This low move helps make room for white to make eyes on the bottom, but it leaves a gap for black to push through."],
  ["C4"],
  ["B5"],
  ["E3"],
  ["F2", "White's G4 group is fairly safe now, but the D4 stone is in trouble."],
  ["D5", "Black captures the white stone. White can't escape by playing at E4."],
  ["E7", "White abandons the D4 stone (for now) and tries to make territory on the top."],
  ["F8", "Blacks wants to expand its territory on the top, while limiting white's."],
  ["E8"],
  ["E9"],
  ["D9"],
  ["F9"],
  ["C8", "The two players peacefully split up the top."],
  ["B4"],
  ["F7"],
  ["G8"],
  ["F5"],
  ["G6"],
  ["J3", "Note that the H4 stones have only two liberties now. This makes the cutting point at h4 very dangerous for black."],
  ["B6", "Instead of defending the cutting point on the right, black sacrifices the H4 stones in order to capture the B5 stone."],
  ["h4", "White completes the trade. The black stones can't escape by playing at J4: white will just capture at J5."],
  ["F4", "Black cuts the white chain in two."],
  ["F3", "Although the D4 stone was left for dead many turns ago, now black is forced to spend a move capturing it. This lets white get a big move elsewhere. Even dead stones are still worth something."],
  ["E4"],
  ["H6"],
  ["F6", "Black captures the F5 stone. White can't get any more liberties by playing at E5."],
  ["B1", "This tricky-looking move makes a big reduction in black's territory. This shape is called a <i>monkey jump</i> (or <i>saru-suberi</i> in Japanese)."],
  ["B2"],
  ["B7"],
  ["C6"],
  ["D7"],
  ["A7"],
  ["A8"],
  ["A6"],
  ["H7", "If you rewind to move 24, it looked like black would make 10 or more points on the right side; white has taken almost all of it away."],
  ["C1"],
  ["E6"],
  ["E5"],
  ["D2"],
  ["C2"],
  ["D1"],
  ["H8"],
  ["J8"],
  ["H9"],
  ["J7"],
  ["D6"],
  ["J9"],
  ["pass"],
  ["pass", "Black has 12 points of territory plus 4 captures. White has 17 points of territory plus 3 captures. So white is already ahead on the board, and wins comfortably after adding komi."],
  ];

  var whiteTerritory = [
      'A9', 'B9', 'C9', 'B8', 'D8',
      'J6', 'J5', 'H4', 'J4',
      'H3',
      'G2', 'J2',
      'E1', 'F1', 'G1', 'H1', 'J1',
  ];
  var blackTerritory = [
      'G9',
      'A5', 'B5', 'C5', 'F5',
      'A4', 'D4',
      'A3', 'B3',
      'A2',
      'A1', 'B1',
  ];


  var BOARD_SIZE = 9;
  var jrecord = new JGO.Record(BOARD_SIZE);
  var jboard = jrecord.jboard;
  var jsetup = new JGO.Setup(jboard, JGO.BOARD.largeWalnut);
  var moveIdx = -1;
  var colnames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N'];
  var lastMove = null;

  function stringToCoords(move_string) {
      var colStr = move_string.substring(0, 1);
      var rowStr = move_string.substring(1);
      var col = colnames.indexOf(colStr);
      var row = BOARD_SIZE - parseInt(rowStr, 10);
      return new JGO.Coordinate(col, row);
  }

  function move(moveDiff) {
      if (moveDiff > 0) {
          var newMoveIdx = Math.min(gameRecord.length - 1, moveIdx + moveDiff);
          while (moveIdx < newMoveIdx) {
              nextMove();
          }
      } else if (moveDiff < 0) {
          var newMoveIdx = Math.max(moveIdx + moveDiff, -1);
          resetGame();
          move(newMoveIdx - moveIdx);
      }
      document.getElementById('move').innerHTML = (moveIdx + 1).toString();
      document.getElementById('moves').innerHTML = gameRecord.length.toString();
  }

  function resetGame() {
      jrecord.jboard.clear();
      jrecord.root = jrecord.current = null;
      jrecord.info = {};
      moveIdx = -1;
      document.getElementById('lastMove').innerHTML = '';
      document.getElementById('comments').innerHTML = '';
  }

  function nextMove() {
      if (moveIdx == gameRecord.length - 1) {
          return;
      }
      moveIdx += 1;
      var player = moveIdx % 2 == 0 ? JGO.BLACK : JGO.WHITE;
      var nextTurn = gameRecord[moveIdx];
      var nextMove = nextTurn[0];
      var comments = nextTurn[1] || '';
      if (nextMove == 'pass') {
          doPass(player);
      } else {
          doStone(player, stringToCoords(nextMove));
      }

      var playerName = player == JGO.BLACK ? 'Black' : 'White';
      var moveText = nextMove == 'pass' ? 'passes' : nextMove;
      document.getElementById('lastMove').innerHTML = playerName + ' ' + moveText;
      document.getElementById('comments').innerHTML = comments;

      if (moveIdx == gameRecord.length - 1) {
          // Show territory.
          var node = jrecord.current;
          for (var i = 0; i < blackTerritory.length; ++i) {
              node.setMark(stringToCoords(blackTerritory[i]), JGO.MARK.SQUARE);
          }
          for (var i = 0; i < whiteTerritory.length; ++i) {
              node.setMark(stringToCoords(whiteTerritory[i]), JGO.MARK.TRIANGLE);
          }
      }
  }

  function doPass(player) {
      // Clear last move mark.
      var node = jrecord.current;
      if (lastMove) {
          node.setMark(lastMove, JGO.MARK.NONE);
      }
      lastMove = null;
  }

  function doStone(player, coords) {
      var play = jboard.playMove(coords, player, false /* ko */);
      if (play.success) {
          var node = jrecord.createNode(true);
          node.info.captures[player] += play.captures.length; // tally captures
          node.setType(coords, player); // play stone
          node.setType(play.captures, JGO.CLEAR); // clear opponent's stones
          if (lastMove) {
              node.setMark(lastMove, JGO.MARK.NONE); // clear previous mark
          }
          node.setMark(coords, JGO.MARK.CIRCLE); // mark move
          lastMove = coords;
      }
  }

  jsetup.create('board', function(canvas) {
      document.body.onkeydown = function(e) {
          if(e.keyCode == 37) move(-1);
          else if(e.keyCode == 39) move(1);
      };
  });
  </script>
  <script type="text/javascript">JGO.auto.init(document, JGO);</script>

  <p>If you want to play a game yourself, you can do this right now. Below is a playable
    demo of a 5x5 Go bot that we built and deployed for you.<p>

  <div style="text-align: center; width: 100%">
    <div style="display: inline-block;">
      <iframe   src="https://www.badukai.com/demos/static/play_mcts_55.html" height="700" width="700" style="border:2px solid grey; background-color: #f8f8f8;"></iframe>
    </div>
  </div>
</div>

## Deep learning and the game of Go

### What machine learning can do for you in Go

Whether you're programming a computer to play Go or tic-tac-toe, most board game AIs share a similar overall structure. Depending on the game, the best solutions may involve game-specific logic, machine learning, or both. Let's have a look at a few tasks you can hope to achieve when attempting to solve board games with a computer:

Selecting moves in the early game: many systems use an opening book, that is a database of opening sequences taken from expert human games.

Searching game states: from the current board state, look ahead and try to evaluate what the outcome might be. Humans mostly use intuition for that, but computers are much stronger at brute forcing computation-heavy solutions. A smart application of these first two points essentially beat Chess world champion Garry Kasparov back in 1997.

Reducing the number of moves to consider: In Go there are around 250 valid moves per turn. This means looking ahead just four moves requires evaluating nearly 4 billion positions. Employing smart heuristics to limit which moves to consider and which to discard immediately might help immensely.

Evaluating game states: If you could perfectly judge how likely a board situation is to win in the end, you'd have a winning strategy: pick the move that maximizes the likelihood of winning the game. Of course, this is a very difficult problem and we can be happy to find good approximations of these values.

In Go, rules-based approaches to move selection turn out to be mediocre at this task: it's extremely difficult to write out rules that reliably identify the most important area of the board. But deep learning is perfectly suited to the problem -- we can apply it to train a computer to imitate a human Go player. Searching and evaluating game states are tasks that deep learning algorithms can be particularly strong at and for the remainder of this article we will focus on predicting expert Go moves.

### Building blocks for a deep learning Go bot

We start with a large collection of game records between strong human players; online gaming servers are a great resource here. Then we replay all the games on a computer, extracting each board position and the following move. That's our training set. With a suitably deep neural network, it's possible to predict the human move with better than 50% accuracy. You can build a bot that just plays the predicted human move, and it's already a credible opponent. Schematically, the application we're trying to build looks as follows:

![Alt text](./img/overview_diagram_sm.jpg)
*How to predict the next move in a game of Go using deep learning.*

On a high level, to build a deep-learning-Go-move-prediction application, you need to address the following tasks:

1. **Downloading and processing Go data:** You can download expert Go data from various Go servers, like KGS. Most of the time this data comes in a specific, text-based format called Smart Go Format (SGF). You need to read game information from such files, for instance how many moves there are, what the current board state or next move is.
2. **Encoding Go data:** Next, the Go board information has to be encoded in a machine-readable way, so that we can feed it into a neural network. The last figure illustrates a simple example of such an encoder, in which black stones are assigned a 1, white stones a -1 and empty points a 0. There are many more sophisticated ways to encode Go games, but this is a good first attempt.
3. **Building and training a deep learning model with Go data:** We then need to build a neural network into which we can feed the encoded Go data. Specifically, we will build a network, to which we show the current board situation and let it predict the next move. This prediction can be compared to the actual outcome (the next move from data). From this comparison, the network will infer how to adapt its parameters to get better at this task -- no explicit programming needed.
4. **Serving the trained model:** Finally, once the model has been built and trained, we need to expose it somewhere for humans or other computers to play against. For humans, a graphical interface is convenient. For bots, you will have to comply with an exchange format like the Go Text Protocol (GTP). At the end of this article, you will run a Go bot served over HTTP with docker.
Explaining all these points in detail in an article is a bit too much, but at least we can show you how to do the third step, namely setting up and training a neural network with DL4J.

## Building and training a deep learning model for Go with Eclipse Deeplearning4J

[Deeplearning4j](https://deeplearning4j.org/) is an open-source deep learning framework at the core of a suite of other [powerful machine learning tools for the JVM](https://github.com/deeplearning4j). Perhaps most notably [ND4J](https://nd4j.org/) provides fast and versatile n-dimensional arrays, which can be used as the foundation for all sorts of [numerical computing on the JVM](https://nd4j.org/).

To build a deep neural network with Deeplearning4J for Go move prediction, we proceed in five steps: loading data, processing data, building a model, training the model and evaluating it - a typical procedure for machine learning applications. We've published the code for this section in the following [GitHub repo](https://github.com/maxpumperla/eclipse_dl4j_go_move_prediction), with instructions to run it with Maven or Docker.

### Loading data

As a first step you need to load data into ND4J arrays. Features and labels for this example have been [created and stored](https://github.com/maxpumperla/eclipse_dl4j_go_move_prediction/tree/master/src/main/resources) for you already, in two separate files. You can load each with a single line of code:

```
INDArray features = Nd4j.createFromNpyFile(new ClassPathResource("features_3000.npy").getFile());
INDArray labels = Nd4j.createFromNpyFile(new ClassPathResource("labels_3000.npy").getFile());
```

### Processing data

Next, you build a so-called DataSet from features and labels, a core ND4J abstraction. Using this DataSet, you can split data into 90% training and 10% test data randomly:

```
        DataSet allData = new DataSet(features, labels);
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.9);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();
 ```
 
### Building a model

With data processed and ready to go, you can now turn to building the actual model that we'll feed the data into. In DL4J you do this by creating a model configuration, a so-called MultiLayerConfiguration. This configuration is built up modularly using a builder pattern, adding properties of the network one by one. Essentially, a neural network configuration consists of general properties for setup and learning and a list of layers. When we feed data into the network, data passes through the network sequentially, layer by layer, until we reach an output or prediction. The following code sets up a neural network tailored towards Go move prediction:

```

    int size = 19;
        int featurePlanes = 11;
        int boardSize = 19 * 19;
        int randomSeed = 1337;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(randomSeed)
                .learningRate(.1)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.ADAGRAD)
                .list()
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(featurePlanes).stride(1, 1).nOut(50).activation(Activation.RELU).build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(1, 1).build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1).nOut(20).activation(Activation.RELU).build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2).stride(1, 1).build())
                .layer(4, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(boardSize).activation(Activation.SOFTMAX).build())
                .setInputType(InputType.convolutional(size, size, featurePlanes))
                .backprop(true).pretrain(false).build();
```

We can't go into details about the individual parts of this model, but note that on a high level it consists of six layers, two convolutional layers, two subsampling layers, a dense layer and an output layer. These layers map our input data to a vector of length 19 times 19, corresponding to the number of possible moves on a 19 by 19 Go board. Convolutional and subsampling layers are often used together and are very good at detecting features in spatial data, such as images, videos, or Go boards.

### Training the model

To train our model, you first need to create a MultiLayerNetwork from the above configuration, initialize it and then fit or train it on the training data:

```
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        model.fit(trainingData);
```

### Evaluating the model

The last step is to evaluate your model on test data and check the results:

```
        Evaluation eval = new Evaluation(19 * 19);
        INDArray output = model.output(testData.getFeatureMatrix());
        eval.eval(testData.getLabels(), output);
        log.info(eval.stats());
```

If you run this yourself, don't be surprised by the relatively poor results of this little experiment. We only use 3000 moves to train the network for a complex game that has about 200 moves per game. So the data we're using is way too small to expect this network to learn to accurately predict next moves. However, the network we chose, trained on a data set consisting of a few hundred thousand games worth of moves, will play Go at an intermediate amateur level. Not bad for just a few lines of code.

## Running an end-to-end deep learning bot

Finally, if you want to run an end-to-end go bot locally on your machine, you can do this quite easily with docker. When pulling the following image from Docker Hub, you have access to all the bots we currently have built for the book:

```
docker pull maxpumperla/dlgo
docker run maxpumperla/dlgo
```

Running these two lines will start up a Python web server that serves several Go bots. For instance, you can play against a 9x9 bot trained with so-called policy gradients, a more advanced deep learning technique, by navigating to the following site in your browser:
```
127.0.0.1:5000/static/play_pg_99.html
```

##Conclusion

We've covered a lot of ground in this article and hope we have sparked your interest in both the field of deep learning and the game of Go. If you want to learn more, check out the following resources:

[Deep Learning: A Practitioner's Approach](https://www.amazon.com/Deep-Learning-Practitioners-Josh-Patterson/dp/1491914254/) This book shows you how to get started with deep learning in a very pragmatic way, giving you an in-depth introduction to Eclipse DL4J and its ecosystem. If you know Java, but are new to deep learning, this might be a very interesting read, and the first four chapters are [freely available](http://go.pardot.com/l/456082/2017-11-29/dtc682).

[Deep Learning and the Game of Go](https://www.manning.com/books/deep-learning-and-the-game-of-go) In this book, you learn all about deep learning as it can be applied in the running example of mastering computer Go. This article contains an excerpt of the first two chapters that are [freely available](https://livebook.manning.com/#!/book/deep-learning-and-the-game-of-go/chapter-1). The online version of the book is packed with interactive tutorials and demos, a few of which you've seen here. Notably, apart from knowing Python at an intermediate level, it does not make any assumptions about your knowledge of machine learning. So, if you're new to this and want an introduction to deep learning by example, this book might be for you (make sure to use "smpumperla40" to save 40%).
