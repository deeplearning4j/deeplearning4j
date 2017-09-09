---
title: 在Android系统中部署Deeplearning4J
layout: cn-default
---
<title>如何在Android应用程序中使用Deeplearning4J</title>
<meta property="og:title" content="如何在Android应用程序中使用Deeplearning4J" />
<meta name="description" content="DeepLearning4J（DL4J）是在JVM上运行的一个热门机器学习库。在本教程中，我将向您介绍如何用它在Android应用程序中创建和训练神经网络。"/>
<meta property="og:description" content="DeepLearning4J（DL4J）是在JVM上运行的一个热门机器学习库。在本教程中，我将向您介绍如何用它在Android应用程序中创建和训练神经网络。"/>
<link rel="canonical" href="http://progur.com/2017/01/how-to-use-deeplearning4j-on-android.html" />
<meta property="og:url" content="http://progur.com/2017/01/how-to-use-deeplearning4j-on-android.html" />
<meta property="og:site_name" content="Progur!"/>
<meta property="og:type" content="article" />
<meta property="article:published_time" content="2017-01-14T00:00:00+05:30" />
<meta name="twitter:card" content="summary" />
<meta name="twitter:site" content="@hathibel" />
<meta name="twitter:creator" content="@hathibel" />
<script type="application/ld+json">
  {
    "@context": "http://schema.org",
    "@type": "BlogPosting",
    "headline": "如何在Android应用程序中使用Deeplearning4J",
    "datePublished": "2017-01-14T00:00:00+05:30",
    "description": "DeepLearning4J（DL4J）是在JVM上运行的一个热门机器学习库。在本教程中，我将向您介绍如何用它在Android应用程序中创建和训练神经网络。",
    "url": "http://progur.com/2017/01/how-to-use-deeplearning4j-on-android.html"
  }
</script>
<!-- End Jekyll SEO tag -->
  <body>

<div class="container">
    <div class="row">
        <div class="col-md-12">
            <div>
                <div>
                    <h1 class="post-title">如何在Android应用程序中使用Deeplearning4J</h1>
                    <div class="post-meta">作者：Ashraff Hathibelagal &bull; 2017年1月7日</div><br>
                    <div class="post-actual-content">
                        <p>一般而言，配有多个GPU的高性能计算机最适合承担训练神经网络的任务。那么，普通的Android手机或平板电脑是否能胜任这项工作呢？这当然是可行的。但是考虑到Android设备的典型配置，运行速度可能会相当缓慢。如果您对此并不在意，请继续阅读。</p>

<p>在本教程中，我将向您介绍如何用<a href="https://github.com/deeplearning4j/deeplearning4j" target="_blank" rel="nofollow">Deeplearning4J</a>这一热门的Java深度学习库来在Android设备上创建和训练神经网络。</p>

<h3 id="prerequisites">必要条件</h3>

<p>为获得最佳结果，您的系统需具备以下条件：</p>

<ul>
  <li>运行21或更高级别API的Android设备或模拟器，至少有大约200 MB的内部存储空间。强烈建议您先使用模拟器，一旦内存或存储空间不足，您可以对模拟器进行快速调整。</li>
  <li>Android Studio 2.2或更新版本</li>
</ul>

<h3 id="configuring-your-android-studio-project">配置Android Studio项目</h3>

<p>为了能在项目中使用Deeplearning4J，请将以下<code class="highlighter-rouge">compile</code>依赖项添加至您的应用模块的<strong>build.gradle</strong>文件中：</p>

<figure class="highlight"><pre><code class="language-groovy" data-lang="groovy"><span class="n">compile</span> <span class="s1">'org.deeplearning4j:deeplearning4j-core:0.7.2'</span>
<span class="n">compile</span> <span class="s1">'org.nd4j:nd4j-native:0.7.2'</span>
<span class="n">compile</span> <span class="s1">'org.nd4j:nd4j-native:0.7.2:android-x86'</span>
<span class="n">compile</span> <span class="s1">'org.nd4j:nd4j-native:0.7.2:android-arm'</span>

</code></pre></figure>

<p>如您所见，DL4J依赖于ND4J（全称为“面向Java的N维数组”）这一可以快速处理N维数组的运算库。ND4J内部依赖一个称为JavaCPP的库，其中包含特定平台的本机代码。因此，您必须加载一个与Android设备的基础系统架构相匹配的ND4J版本。我用的是一台x86设备，所以我的平台是<code class="highlighter-rouge">android-x86</code>。</p>

<p>DL4J和ND4J有多个依赖项文件的名字相同。为了避免构建错误，请将以下<code class="highlighter-rouge">exclude</code>参数添加至您的<code class="highlighter-rouge">packagingOptions</code>。</p>

<figure class="highlight"><pre><code class="language-groovy" data-lang="groovy"><span class="n">packagingOptions</span> <span class="o">{</span>
    <span class="n">exclude</span> <span class="s1">'META-INF/DEPENDENCIES'</span>
    <span class="n">exclude</span> <span class="s1">'META-INF/DEPENDENCIES.txt'</span>
    <span class="n">exclude</span> <span class="s1">'META-INF/LICENSE'</span>
    <span class="n">exclude</span> <span class="s1">'META-INF/LICENSE.txt'</span>
    <span class="n">exclude</span> <span class="s1">'META-INF/license.txt'</span>
    <span class="n">exclude</span> <span class="s1">'META-INF/NOTICE'</span>
    <span class="n">exclude</span> <span class="s1">'META-INF/NOTICE.txt'</span>
    <span class="n">exclude</span> <span class="s1">'META-INF/notice.txt'</span>
    <span class="n">exclude</span> <span class="s1">'META-INF/INDEX.LIST'</span>
<span class="o">}</span></code></pre></figure>

<p>此外，编译完成后的代码中的方法数量将远远超过65,536。应对方法是在<code class="highlighter-rouge">defaultConfig</code>中添加以下选项：</p>

<figure class="highlight"><pre><code class="language-groovy" data-lang="groovy"><span class="n">multiDexEnabled</span> <span class="kc">true</span></code></pre></figure>

<p>然后请点击<strong>Sync Now</strong>，更新项目。</p>

<h3 id="starting-an-asynchronous-task">启动一项异步任务</h3>

<p>训练神经网络需要消耗大量CPU资源，因此最好不要在应用程序的UI线程中运行。我不太确定DL4J是否默认以异步方式训练网络。保险起见，我会先用<code class="highlighter-rouge">AsyncTask</code>类来生成一个独立的线程。</p>

<figure class="highlight"><pre><code class="language-java" data-lang="java"><span class="n">AsyncTask</span><span class="o">.</span><span class="na">execute</span><span class="o">(</span><span class="k">new</span> <span class="n">Runnable</span><span class="o">()</span> <span class="o">{</span>
    <span class="nd">@Override</span>
    <span class="kd">public</span> <span class="kt">void</span> <span class="nf">run</span><span class="o">()</span> <span class="o">{</span>
        <span class="n">createAndUseNetwork</span><span class="o">();</span>
    <span class="o">}</span>
<span class="o">});</span></code></pre></figure>

<p><code class="highlighter-rouge">createAndUseNetwork()</code>方法还不存在，需要自行创建。</p>

<figure class="highlight"><pre><code class="language-java" data-lang="java"><span class="kd">private</span> <span class="kt">void</span> <span class="nf">createAndUseNetwork</span><span class="o">()</span> <span class="o">{</span>

<span class="o">}</span></code></pre></figure>

<h3 id="creating-a-neural-network">创建神经网络</h3>

<p>DL4J的API非常直观易用。现在让我们用它来创建一个具有隐藏层的简单的多层感知器。网络将接受两项输入值，产生一项输出值。我们用<code class="highlighter-rouge">DenseLayer</code>（稠密层）和<code class="highlighter-rouge">OutputLayer</code>（输出层）两个类来创建网络的层。请将以下代码添加至您前一步创建的<code class="highlighter-rouge">createAndUseNetwork()</code>方法中：</p>

<figure class="highlight"><pre><code class="language-java" data-lang="java"><span class="n">DenseLayer</span> <span class="n">inputLayer</span> <span class="o">=</span> <span class="k">new</span> <span class="n">DenseLayer</span><span class="o">.</span><span class="na">Builder</span><span class="o">()</span>
        <span class="o">.</span><span class="na">nIn</span><span class="o">(</span><span class="mi">2</span><span class="o">)</span>
        <span class="o">.</span><span class="na">nOut</span><span class="o">(</span><span class="mi">3</span><span class="o">)</span>
        <span class="o">.</span><span class="na">name</span><span class="o">(</span><span class="s">"Input"</span><span class="o">)</span>
        <span class="o">.</span><span class="na">build</span><span class="o">();</span>

<span class="n">DenseLayer</span> <span class="n">hiddenLayer</span> <span class="o">=</span> <span class="k">new</span> <span class="n">DenseLayer</span><span class="o">.</span><span class="na">Builder</span><span class="o">()</span>
        <span class="o">.</span><span class="na">nIn</span><span class="o">(</span><span class="mi">3</span><span class="o">)</span>
        <span class="o">.</span><span class="na">nOut</span><span class="o">(</span><span class="mi">2</span><span class="o">)</span>
        <span class="o">.</span><span class="na">name</span><span class="o">(</span><span class="s">"Hidden"</span><span class="o">)</span>
        <span class="o">.</span><span class="na">build</span><span class="o">();</span>

<span class="n">OutputLayer</span> <span class="n">outputLayer</span> <span class="o">=</span> <span class="k">new</span> <span class="n">OutputLayer</span><span class="o">.</span><span class="na">Builder</span><span class="o">()</span>
        <span class="o">.</span><span class="na">nIn</span><span class="o">(</span><span class="mi">2</span><span class="o">)</span>
        <span class="o">.</span><span class="na">nOut</span><span class="o">(</span><span class="mi">1</span><span class="o">)</span>
        <span class="o">.</span><span class="na">name</span><span class="o">(</span><span class="s">"Output"</span><span class="o">)</span>
        <span class="o">.</span><span class="na">build</span><span class="o">();</span></code></pre></figure>

<p>现在层已经准备就绪，可以创建一个<code class="highlighter-rouge">NeuralNetConfiguration.Builder</code>对象来配置我们的神经网络。</p>

<figure class="highlight"><pre><code class="language-java" data-lang="java"><span class="n">NeuralNetConfiguration</span><span class="o">.</span><span class="na">Builder</span> <span class="n">nncBuilder</span> <span class="o">=</span> <span class="k">new</span> <span class="n">NeuralNetConfiguration</span><span class="o">.</span><span class="na">Builder</span><span class="o">();</span>
<span class="n">nncBuilder</span><span class="o">.</span><span class="na">iterations</span><span class="o">(</span><span class="mi">10000</span><span class="o">);</span>
<span class="n">nncBuilder</span><span class="o">.</span><span class="na">learningRate</span><span class="o">(</span><span class="mf">0.01</span><span class="o">);</span></code></pre></figure>

<p>在上述代码中，我设置了两项重要参数的值：学习速率和迭代次数。您可以自行改变这些值。</p>

<p>现在我们必须创建一个<code class="highlighter-rouge">NeuralNetConfiguration.ListBuilder</code>对象来将层连接起来，为其明确指令。</p>

<figure class="highlight"><pre><code class="language-java" data-lang="java"><span class="n">NeuralNetConfiguration</span><span class="o">.</span><span class="na">ListBuilder</span> <span class="n">listBuilder</span> <span class="o">=</span> <span class="n">nncBuilder</span><span class="o">.</span><span class="na">list</span><span class="o">();</span>
<span class="n">listBuilder</span><span class="o">.</span><span class="na">layer</span><span class="o">(</span><span class="mi">0</span><span class="o">,</span> <span class="n">inputLayer</span><span class="o">);</span>
<span class="n">listBuilder</span><span class="o">.</span><span class="na">layer</span><span class="o">(</span><span class="mi">1</span><span class="o">,</span> <span class="n">hiddenLayer</span><span class="o">);</span>
<span class="n">listBuilder</span><span class="o">.</span><span class="na">layer</span><span class="o">(</span><span class="mi">2</span><span class="o">,</span> <span class="n">outputLayer</span><span class="o">);</span></code></pre></figure>

<p>然后用以下代码启用反向传播：</p>

<figure class="highlight"><pre><code class="language-java" data-lang="java"><span class="n">listBuilder</span><span class="o">.</span><span class="na">backprop</span><span class="o">(</span><span class="kc">true</span><span class="o">);</span></code></pre></figure>

<p>至此，我们的神经网络就可以作为<code class="highlighter-rouge">MultiLayerNetwork</code>类的一个实例生成并初始化了。</p>

<figure class="highlight"><pre><code class="language-java" data-lang="java"><span class="n">MultiLayerNetwork</span> <span class="n">myNetwork</span> <span class="o">=</span> <span class="k">new</span> <span class="n">MultiLayerNetwork</span><span class="o">(</span><span class="n">listBuilder</span><span class="o">.</span><span class="na">build</span><span class="o">());</span>
<span class="n">myNetwork</span><span class="o">.</span><span class="na">init</span><span class="o">();</span></code></pre></figure>

<h3 id="creating-training-data">创建训练数据</h3>

<p>为了创建训练数据，我们要用到ND4J提供的<code class="highlighter-rouge">INDArray</code>类。训练数据的形式如下：</p>

<div class="highlighter-rouge"><pre class="highlight"><code>输入      预期输出
------      ----------------
0,0         0
0,1         1
1,0         1
1,1         0
</code></pre>
</div>

<p>您可能已经猜到，我们的神经网络的运作模式将会和异或门一样。训练数据包含四个样例，必须在代码中提及。</p>

<figure class="highlight"><pre><code class="language-java" data-lang="java"><span class="kd">final</span> <span class="kt">int</span> <span class="n">NUM_SAMPLES</span> <span class="o">=</span> <span class="mi">4</span><span class="o">;</span></code></pre></figure>

<p>接下来为输入和预期输出创建两个<code class="highlighter-rouge">INDArray</code>对象，并将其初始化为零。</p>

<figure class="highlight"><pre><code class="language-java" data-lang="java"><span class="n">INDArray</span> <span class="n">trainingInputs</span> <span class="o">=</span> <span class="n">Nd4j</span><span class="o">.</span><span class="na">zeros</span><span class="o">(</span><span class="n">NUM_SAMPLES</span><span class="o">,</span> <span class="n">inputLayer</span><span class="o">.</span><span class="na">getNIn</span><span class="o">());</span>
<span class="n">INDArray</span> <span class="n">trainingOutputs</span> <span class="o">=</span> <span class="n">Nd4j</span><span class="o">.</span><span class="na">zeros</span><span class="o">(</span><span class="n">NUM_SAMPLES</span><span class="o">,</span> <span class="n">outputLayer</span><span class="o">.</span><span class="na">getNOut</span><span class="o">());</span></code></pre></figure>

<p>请注意，输入数组中的列数应等于输入层中的神经元数量。与此类似，输出数组中的列数应等于输出层中的神经元数量。</p>

<p>用训练数据填充这些数组很容易。只需使用<code class="highlighter-rouge">putScalar()</code>方法即可：</p>

<figure class="highlight"><pre><code class="language-java" data-lang="java"><span class="c1">// 如输入0,0则显示0</span>
<span class="n">trainingInputs</span><span class="o">.</span><span class="na">putScalar</span><span class="o">(</span><span class="k">new</span> <span class="kt">int</span><span class="o">[]{</span><span class="mi">0</span><span class="o">,</span><span class="mi">0</span><span class="o">},</span> <span class="mi">0</span><span class="o">);</span>
<span class="n">trainingInputs</span><span class="o">.</span><span class="na">putScalar</span><span class="o">(</span><span class="k">new</span> <span class="kt">int</span><span class="o">[]{</span><span class="mi">0</span><span class="o">,</span><span class="mi">1</span><span class="o">},</span> <span class="mi">0</span><span class="o">);</span>
<span class="n">trainingOutputs</span><span class="o">.</span><span class="na">putScalar</span><span class="o">(</span><span class="k">new</span> <span class="kt">int</span><span class="o">[]{</span><span class="mi">0</span><span class="o">,</span><span class="mi">0</span><span class="o">},</span> <span class="mi">0</span><span class="o">);</span>

<span class="c1">// 如输入0,1则显示1</span>
<span class="n">trainingInputs</span><span class="o">.</span><span class="na">putScalar</span><span class="o">(</span><span class="k">new</span> <span class="kt">int</span><span class="o">[]{</span><span class="mi">1</span><span class="o">,</span><span class="mi">0</span><span class="o">},</span> <span class="mi">0</span><span class="o">);</span>
<span class="n">trainingInputs</span><span class="o">.</span><span class="na">putScalar</span><span class="o">(</span><span class="k">new</span> <span class="kt">int</span><span class="o">[]{</span><span class="mi">1</span><span class="o">,</span><span class="mi">1</span><span class="o">},</span> <span class="mi">1</span><span class="o">);</span>
<span class="n">trainingOutputs</span><span class="o">.</span><span class="na">putScalar</span><span class="o">(</span><span class="k">new</span> <span class="kt">int</span><span class="o">[]{</span><span class="mi">1</span><span class="o">,</span><span class="mi">0</span><span class="o">},</span> <span class="mi">1</span><span class="o">);</span>

<span class="c1">// 如输入1,0则显示1</span>
<span class="n">trainingInputs</span><span class="o">.</span><span class="na">putScalar</span><span class="o">(</span><span class="k">new</span> <span class="kt">int</span><span class="o">[]{</span><span class="mi">2</span><span class="o">,</span><span class="mi">0</span><span class="o">},</span> <span class="mi">1</span><span class="o">);</span>
<span class="n">trainingInputs</span><span class="o">.</span><span class="na">putScalar</span><span class="o">(</span><span class="k">new</span> <span class="kt">int</span><span class="o">[]{</span><span class="mi">2</span><span class="o">,</span><span class="mi">1</span><span class="o">},</span> <span class="mi">0</span><span class="o">);</span>
<span class="n">trainingOutputs</span><span class="o">.</span><span class="na">putScalar</span><span class="o">(</span><span class="k">new</span> <span class="kt">int</span><span class="o">[]{</span><span class="mi">2</span><span class="o">,</span><span class="mi">0</span><span class="o">},</span> <span class="mi">1</span><span class="o">);</span>

<span class="c1">// 如输入1,1则显示0</span>
<span class="n">trainingInputs</span><span class="o">.</span><span class="na">putScalar</span><span class="o">(</span><span class="k">new</span> <span class="kt">int</span><span class="o">[]{</span><span class="mi">3</span><span class="o">,</span><span class="mi">0</span><span class="o">},</span> <span class="mi">1</span><span class="o">);</span>
<span class="n">trainingInputs</span><span class="o">.</span><span class="na">putScalar</span><span class="o">(</span><span class="k">new</span> <span class="kt">int</span><span class="o">[]{</span><span class="mi">3</span><span class="o">,</span><span class="mi">1</span><span class="o">},</span> <span class="mi">1</span><span class="o">);</span>
<span class="n">trainingOutputs</span><span class="o">.</span><span class="na">putScalar</span><span class="o">(</span><span class="k">new</span> <span class="kt">int</span><span class="o">[]{</span><span class="mi">3</span><span class="o">,</span><span class="mi">0</span><span class="o">},</span> <span class="mi">0</span><span class="o">);</span></code></pre></figure>

<p>我们不会直接使用<code class="highlighter-rouge">INDArray</code>对象，而是将其转换为一个<code class="highlighter-rouge">DataSet</code>。</p>

<figure class="highlight"><pre><code class="language-java" data-lang="java"><span class="n">DataSet</span> <span class="n">myData</span> <span class="o">=</span> <span class="k">new</span> <span class="n">DataSet</span><span class="o">(</span><span class="n">trainingInputs</span><span class="o">,</span> <span class="n">trainingOutputs</span><span class="o">);</span></code></pre></figure>

<p>至此，我们可以调用<code class="highlighter-rouge">fit()</code>方法，将数据集输入神经网络，开始训练。</p>

<figure class="highlight"><pre><code class="language-java" data-lang="java"><span class="n">myNetwork</span><span class="o">.</span><span class="na">fit</span><span class="o">(</span><span class="n">myData</span><span class="o">);</span></code></pre></figure>

<p>以上就是全部的流程。您的神经网络已经准备就绪，可以投入使用了。</p>

<h3 id="conclusion">总结</h3>

<P>本教程已向您说明，在Android Studio项目中用Deeplearning4J学习库来创建和训练神经网络是非常容易的。但是我要提醒您，在某些情况下，用电池驱动的低性能设备来训练神经网络可能并不是个好主意。</p>

<p>本文最初由Ashraff Hathibelagal发表于<a href="http://progur.com/2017/01/how-to-use-deeplearning4j-on-android.html" target="_blank" rel="nofollow">Progur</a>。

                

<footer class="footer">
  <div class="container">
  </div>
</footer>
  
