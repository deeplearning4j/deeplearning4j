<!DOCTYPE html>

<html lang="en">
<head>
    <meta charset="utf-8">

    <title>Word2Vec Searcher</title>
    <!-- jQuery -->
    <script src="//ajax.googleapis.com/ajax/libs/jquery/1/jquery.min.js"></script>
    <script src="assets/views.js"></script>
    <script src="assets/jquery.rest.min.js"></script>

    <script></script>
</head>

<body>
<div>
    <h1>Note that in any and all cases, you can only use words that occur in the vocab. Anything else would require more data.</h1>
</div>


<div>
    <h5>Number of words in vocab.</h5>
   <div><label id="numwords" /></div>
</div>


<div>
    <h5>Word Frequency - Enter a word and hit submit, and get the count of the word in the vocab.</h5>
    <form id ="word_frequency">
        <fieldset>
            <label for="word">Word</label> <input name="word" />


        </fieldset>
        <button id="word_freq_button" value="submit">Submit</button>
        <div  id="word_freq_display" />
    </form>

</div>


f

<div>
<h5>Similarity (distance between 2 word vectors)</h5>
<form id="similarity">
    <fieldset>
        <label for="w1">Word 1</label><input name="w1" value="First word" />
        <label for="w2">Word 1</label> <input name="w2" value="Second word" />
    </fieldset>
    <button  id="sim_submit" value="submit">Submit</button>
    <h5>Similarity outut</h5>
    <div id="simdisplay" />
</form>
</div>

<div>
    <h5>Clustering: What topic the words belongs to</h5>
    <form id="cluster">
        <fieldset>
            <label for="word">Classify</label> <input name="classify_word" id="classify_word"/>

        </fieldset>
        <button id="classify_submit" value="submit">Submit</button>
    </form>
    <h5>Cluster the word belongs to display</h5>
    <div id="cluster_display" />
</div>

<div>
    <h5>Words nearest</h5>
    <form id="nearest">
        <fieldset>
            <label for="word">Word</label> <input name="word" id="word"/>
            <label for="nearest_num">Number</label> <input name="nearest_num" id="nearest_num"/>
        </fieldset>
        <button id="nearest_submit" value="submit">Submit</button>
    </form>
    <h5>Cluster the word belongs to display</h5>
    <div id="nearest_display" />
</div>



<div>
    <h5>Search</h5>
    <form id="search">
        <fieldset>
            <label for="search_query">Search Query</label> <input name="search_query" id="search_query"/>
            <label for="num_results">Number of results</label> <input name="num_results" id="num_results"/>
        </fieldset>
        <button id="search_submit" value="submit">Submit</button>
    </form>
    <h5>Search Results</h5>
    <div id="search_display" />
</div>


</body>
</html>