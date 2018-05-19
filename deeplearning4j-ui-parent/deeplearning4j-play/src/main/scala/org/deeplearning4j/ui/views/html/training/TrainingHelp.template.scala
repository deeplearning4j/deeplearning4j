
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingHelp_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingHelp extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply/*1.2*/(i18n: org.deeplearning4j.ui.api.I18N):play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.40*/("""
"""),format.raw/*2.1*/("""<!DOCTYPE html>
<html lang="en">
    <head>

        <meta charset="utf-8">
        <title>"""),_display_(/*7.17*/i18n/*7.21*/.getMessage("train.pagetitle")),format.raw/*7.51*/("""</title>
            <!-- Start Mobile Specific -->
        <meta name="viewport" content="width=device-width, initial-scale=1">
            <!-- End Mobile Specific -->

        <link id="bootstrap-style" href="/assets/css/bootstrap.min.css" rel="stylesheet">
        <link href="/assets/css/bootstrap-responsive.min.css" rel="stylesheet">
        <link id="base-style" href="/assets/css/style.css" rel="stylesheet">
        <link id="base-style-responsive" href="/assets/css/style-responsive.css" rel="stylesheet">
        <link href='/assets/css/opensans-fonts.css' rel='stylesheet' type='text/css'>
        <link rel="shortcut icon" href="/assets/img/favicon.ico">

            <!-- The HTML5 shim, for IE6-8 support of HTML5 elements -->
            <!--[if lt IE 9]>
	  	<script src="http://html5shim.googlecode.com/svn/trunk/html5.js"></script>
		<link id="ie-style" href="/assets/css/ie.css" rel="stylesheet"/>
	<![endif]-->

            <!--[if IE 9]>
		<link id="ie9style" href="/assets/css/ie9.css" rel="stylesheet"/>
	<![endif]-->

    </head>

    <body>
            <!-- Start Header -->
        <div class="navbar">
            <div class="navbar-inner">
                <div class="container-fluid">
                    <a class="btn btn-navbar" data-toggle="collapse" data-target=".top-nav.nav-collapse,.sidebar-nav.nav-collapse">
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                        <span class="icon-bar"></span>
                    </a>
                    <a class="brand" href="#"><span>"""),_display_(/*41.54*/i18n/*41.58*/.getMessage("train.pagetitle")),format.raw/*41.88*/("""</span></a>
                </div>
            </div>
        </div>
            <!-- End Header -->

        <div class="container-fluid-full">
            <div class="row-fluid">

                    <!-- Start Main Menu -->
                <div id="sidebar-left" class="span2">
                    <div class="nav-collapse sidebar-nav">
                        <ul class="nav nav-tabs nav-stacked main-menu">
                            <li><a href="overview"><i class="icon-bar-chart"></i><span class="hidden-tablet"> """),_display_(/*54.112*/i18n/*54.116*/.getMessage("train.nav.overview")),format.raw/*54.149*/("""</span></a></li>
                            <li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet"> """),_display_(/*55.105*/i18n/*55.109*/.getMessage("train.nav.model")),format.raw/*55.139*/("""</span></a></li>
                            <li><a href="system"><i class="icon-dashboard"></i><span class="hidden-tablet"> """),_display_(/*56.110*/i18n/*56.114*/.getMessage("train.nav.system")),format.raw/*56.145*/("""</span></a></li>
                            <li class="active"><a href="javascript:void(0);"><i class="icon-star"></i><span class="hidden-tablet"> """),_display_(/*57.133*/i18n/*57.137*/.getMessage("train.nav.userguide")),format.raw/*57.171*/("""</span></a></li>
                            <li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet">"""),_display_(/*59.146*/i18n/*59.150*/.getMessage("train.nav.language")),format.raw/*59.183*/("""</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'help')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> English</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ja', 'help')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 日本語</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('zh', 'help')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 中文</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ko', 'help')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 한글</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ru', 'help')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> русский</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('uk', 'help')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> український</span></a></li>
                                </ul>
                            </li>
                        </ul>
                    </div>
                </div>
                    <!-- End Main Menu -->

                <noscript>
                    <div class="alert alert-block span10">
                        <h4 class="alert-heading">Warning!</h4>
                        <p>You need to have <a href="http://en.wikipedia.org/wiki/JavaScript" target="_blank">
                            JavaScript</a> enabled to use this site.</p>
                    </div>
                </noscript>

                    <!-- Start Content -->
                <div id="content" class="span10">
                        <!-- Begin User Guide -->
                    <div class="row-fluid">
                        <div class="box span9">
                            <div class="box-header">
                                <h2><b>User Guide</b></h2>
                            </div>
                            <div class="box-content">

                                <div class="page-header">
                                    <h1>Deeplearning4j <small>Training UI</small></h1>
                                </div>

                                <div class="row-fluid">
                                    <div class="span8">
                                        <h1><small>Welcome!</small></h1>
                                        <p>
                                            Welcome to the Deeplearning4j Training UI! DL4J provides the HistogramIterationListener as a method of visualizing in your browser (in real time) the progress of network training. Here’s an excellent <a href="https://cs231n.github.io/neural-networks-3/#baby" target="_blank">
                                            web page by Andrej Karpathy</a>
                                            about visualizing neural net training. It is worth reading that page first.
                                        </p>
                                    </div>

                                    <div class="span4">
                                        <div class="well">
                                            <h1><small>
                                                If there's any confusion, please ask our engineers in <a href="https://gitter.im/deeplearning4j/deeplearning4j" target="_blank">
                                                Gitter</a>.</small></h1>
                                        </div>
                                    </div>
                                </div>

                                <h1><small>Overview Tab</small></h1>
                                <div class="row-fluid">
                                    <div class="span6">
                                        <h2>Score vs Iteration: Snapshot</h2>
                                        <ul>
                                            <li>Score vs. iteration should (overall) go down over time.</li>
                                            <ul>
                                                <li>If the score increases consistently, your learning rate is likely set too high. Try reducing it until scores become more stable.</li>
                                                <li>Increasing scores can also be indicative of other network issues, such as incorrect data normalization.</li>
                                                <li>If the score is flat or decreases very slowly (over a few hundred iteratons) (a) your learning rate may be too low, or (b) you might be having diffulties with optimization. In the latter case, if you are using the SGD updater, try a different updater such as momentum, RMSProp or Adagrad.</li>
                                                <li>Note that data that isn’t shuffled (i.e., each minibatch contains only one class, for classification) can result in very rough or abnormal-looking score vs. iteration graphs.</li>
                                            </ul>
                                            <li>Some noise in this line chart is expected (i.e., the line will go up and down within a small range). However, if the scores vary quite significantly between runs variation is very large, this can be a problem.</li>
                                        </ul>
                                    </div>
                                    <div class="span6">
                                        <h2>Model Performance</h2>
                                        <p>
                                            The table contains basic model performance metrics.<br><br>
                                            <b>Model Type</b> - MultiLayerNetwork or...<br>
                                            <b>nLayers</b> - Number of layers.<br>
                                            <b>nParams</b> - Number of parameters.<br>
                                            <b>Total Runtime</b> - Explain importance<br>
                                            <b>Last Update</b> - Explain importance<br>
                                            <b>Total Parameter Updates</b> - Explain importance<br>
                                            <b>Updates Per Second</b> - Explain importance<br>
                                            <b>Examples Per Second</b> - Explain importance
                                        </p>
                                    </div>
                                </div>

                                <div class="row-fluid">
                                    <div class="span6">
                                        <h2>Ratio of Updates to Parameters: All Layers</h2>
                                        <p>
                                            Need Explanation Here.
                                        </p>
                                    </div>
                                    <div class="span6">
                                        <h2>Variances</h2>
                                        <p>
                                            Need Explanation Here.
                                        </p>
                                    </div>
                                </div>

                                <h1><small>Model Tab</small></h1>
                                <div class="row-fluid">
                                    <div class="span6">
                                        <h2>Layer Visualization UI</h2>
                                        <p>
                                            The layer visualization UI renders network structure dynamically. Users can inspect the and node layer parameters by clicking on the various elements of the GUI to see general information about layers/nodes, overall network information such as performance.
                                        </p>
                                    </div>
                                    <div class="span6">
                                        <h2>Layer Information</h2>
                                        <p>
                                            The table contains basic layer information.<br><br>
                                            <b>Name</b> - MultiLayerNetwork or...<br>
                                            <b>Type</b> - Number of layers.<br>
                                            <b>Inputs</b> - Number of parameters.<br>
                                            <b>Outputs</b> - Explain importance<br>
                                            <b>Activation Function</b> - Explain importance<br>
                                            <b>Learning Rate</b> - Explain importance
                                        </p>
                                    </div>
                                </div>

                                <div class="row-fluid">
                                    <div class="span6">
                                        <h2>Mean Magnitudes</h2>
                                        <p>
                                    <ul>
                                        <li>At the right is a line chart of the mean magnitude of both the parameters and the updates in the neural network.</li>
                                        <ul>
                                            <li>“Mean magnitude” = the average of the absolute value of the parameters or updates.</li>
                                        </ul>
                                        <li>For tuning the learning rate, the ratio of parameters to updates for a layer should be somewhere in the order of 1000:1 - but note that is a rough guide only, and may not be appropriate for all networks. It’s often a good starting point, however.</li>
                                        <ul>
                                            <li>If the ratio diverges significantly from this, your parameters may be too unstable to learn useful features, or may change too slowly to learn useful features</li>
                                            <li>To change this ratio, adjust your learning rate (or sometimes, parameter initialization). In some networks, you may need to set the learning rate differently for different layers.</li>
                                        </ul>
                                        <li>Keep an eye out for unusually large spikes in the updates: this may indicate exploding gradients (see discussion in the “histogram of gradients” section above)</li>
                                        </p>
                                    </div>
                                    <div class="span6">
                                        <h2>Activations</h2>
                                        <p>
                                            Need Explanation Here.
                                        </p>
                                    </div>
                                </div>

                                <div class="row-fluid">
                                    <div class="span6">
                                        <h2>Parameters Histogram</h2>
                                        <p>
                                    <ul>
                                        <li>At the top right is a histogram of the weights in the neural network (at the last iteration), split up by layer and the type of parameter. For example, “param_0_W” refers to the weight parameters for the first layer.</li>
                                        <li>For weights, these histograms should have an approximately Gaussian (normal) distribution, after some time.</li>
                                        <li>For biases, these histograms will generally start at 0, and will usually end up being approximately Gaussian.</li>
                                        <ul>
                                            <li>One exception to this is for LSTM recurrent neural network layers: by default, the biases for one gate (the forget gate) are set to 1.0 (by default, though this is configurable), to help in learning dependencies across long time periods. This results in the bias graphs initially having many biases around 0.0, with another set of biases around 1.0</li>
                                        </ul>
                                        <li>Keep an eye out for parameters that are diverging to +/- infinity: this may be due to too high a learning rate, or insufficient regularization (try adding some L2 regularization to your network).</li>
                                        <li>Keep an eye out for biases that become very large. This can sometimes occur in the output layer for classification, if the distribution of classes is very imbalanced</li>
                                    </ul>
                                        </p>
                                    </div>
                                    <div class="span6">
                                        <h2>Updates Histogram</h2>
                                        <p>
                                    <ul>
                                        <li>At the bottom left is the histogram of updates for the neural network (at the last iteration), also split up by layer and type of parameter.</li>
                                        <ul>
                                            <li>Note that these are the updates - i.e., the gradients after appling learning rate, momentum, regularization etc.</li>
                                        </ul>
                                        <li>As with the parameter graphs, these should have an approximately Gaussian (normal) distribution.</li>
                                        <li>Keep an eye out for very large values: this can indicate exploding gradients in your network.</li>
                                        <ul>
                                            <li>Exploding gradients are problematic as they can ‘mess up’ the parameters of your network.</li>
                                            <li>In this case, it may indicate a weight initialization, learning rate or input/labels data normalization issue.</li>
                                            <li>In the case of recurrent neural networks, adding some gradient normalization or gradient clipping can frequently help.</li>
                                        </ul>
                                    </ul>
                                        </p>
                                    </div>
                                </div>

                                <h1><small>System Tab</small></h1>
                                <div class="row-fluid">
                                    <div class="span6">
                                        <h2>JVM Memory Utilization</h2>
                                        <p>
                                            Need Explanation Here.
                                        </p>
                                    </div>
                                    <div class="span6">
                                        <h2>Off-Heap Memory Utilization</h2>
                                        <p>
                                            Need Explanation Here.
                                        </p>
                                    </div>
                                </div>
                                <div class="row-fluid">
                                    <div class="span6">
                                        <h2>Hardware Information</h2>
                                        <p>
                                            The table contains basic hardware metrics.<br><br>
                                            <b>JVM Current Memory</b> - What this means.<br>
                                            <b>JVM Max Memory</b> - What this means.<br>
                                            <b>Off-Heap Current Memory</b> - What this means.<br>
                                            <b>Off-Heap Current Memory</b> - What this means.<br>
                                            <b>JVM Available Processors</b> - What this means.<br>
                                            <b>Number Compute Devices</b> - What this means.
                                        </p>
                                    </div>
                                    <div class="span6">
                                        <h2>Software Information</h2>
                                        <p>
                                            The table contains basic software information.<br><br>
                                            <b>OS</b> - What this means.<br>
                                            <b>Host Name</b> - What this means.<br>
                                            <b>OS Architecture</b> - What this means.<br>
                                            <b>JVM Name</b> - What this means.<br>
                                            <b>JVM Version</b> - What this means.<br>
                                            <b>ND4J Backend</b> - What this means.<br>
                                            <b>ND4J Datatype</b> - What this means.
                                        </p>
                                    </div>
                                </div>
                                <div class="row-fluid">
                                    <div class="span6">
                                        <h2>GPU Specific Graph?</h2>
                                        <p>
                                            Need Explanation Here.
                                        </p>
                                    </div>
                                    <div class="span6">
                                        <h2>GPU Specific Table?</h2>
                                        <p>
                                            Need Explanation Here.
                                        </p>
                                    </div>
                                </div>

                            </div>
                        </div>
                            <!-- End User Guide -->
                            <!-- Begin Table of Contents -->
                        <div class="box span3">
                            <div class="box-header">
                                <h2><b>Table of Contents</b></h2>
                            </div>
                            <div class="box-content">
                                <dl>
                                    <dt>Overview</dt>
                                    <dd>Snapshot of your model performance.</dd>
                                    <dt>Model</dt>
                                    <dd>Layer by layer inspection tool.</dd>
                                    <dt>System</dt>
                                    <dd>Memory utilization dashboard as well as system configurations across multiple machines.</dd>
                                    <dt>Language</dt>
                                    <dd>Switch between English, Japanese, Chinese, Korean, Ukranian and Russian.</dd>
                                </dl>
                            </div>
                        </div><!-- End Table of Contents -->
                    </div><!-- End Row Fluid -->
                </div>
          """),
format.raw("""          <!-- End Content -->
            </div><!--End Row Fluid -->
        </div><!-- End Container Fluid Full-->

        <!-- Start JavaScript-->
        <script src="/assets/js/jquery-1.9.1.min.js"></script>
        <script src="/assets/js/jquery-migrate-1.0.0.min.js"></script>
        <script src="/assets/js/jquery-ui-1.10.0.custom.min.js"></script>
        <script src="/assets/js/jquery.ui.touch-punch.js"></script>
        <script src="/assets/js/modernizr.js"></script>
        <script src="/assets/js/bootstrap.min.js"></script>
        <script src="/assets/js/jquery.cookie.js"></script>
        <script src="/assets/js/fullcalendar.min.js"></script>
        <script src="/assets/js/jquery.dataTables.min.js"></script>
        <script src="/assets/js/excanvas.js"></script>
        <script src="/assets/js/jquery.flot.js"></script>
        <script src="/assets/js/jquery.flot.pie.js"></script>
        <script src="/assets/js/jquery.flot.stack.js"></script>
        <script src="/assets/js/jquery.flot.resize.min.js"></script>
        <script src="/assets/js/jquery.chosen.min.js"></script>
        <script src="/assets/js/jquery.uniform.min.js"></script>
        <script src="/assets/js/jquery.cleditor.min.js"></script>
        <script src="/assets/js/jquery.noty.js"></script>
        <script src="/assets/js/jquery.elfinder.min.js"></script>
        <script src="/assets/js/jquery.raty.min.js"></script>
        <script src="/assets/js/jquery.iphone.toggle.js"></script>
        <script src="/assets/js/jquery.uploadify-3.1.min.js"></script>
        <script src="/assets/js/jquery.gritter.min.js"></script>
        <script src="/assets/js/jquery.imagesloaded.js"></script>
        <script src="/assets/js/jquery.masonry.min.js"></script>
        <script src="/assets/js/jquery.knob.modified.js"></script>
        <script src="/assets/js/jquery.sparkline.min.js"></script>
        <script src="/assets/js/counter.js"></script>
        <script src="/assets/js/retina.js"></script>
        <script src="/assets/js/train/train.js"></script>   <!-- Common (lang selection, etc) -->
        <!-- End JavaScript-->

    </body>
</html>
"""))
      }
    }
  }

  def render(i18n:org.deeplearning4j.ui.api.I18N): play.twirl.api.HtmlFormat.Appendable = apply(i18n)

  def f:((org.deeplearning4j.ui.api.I18N) => play.twirl.api.HtmlFormat.Appendable) = (i18n) => apply(i18n)

  def ref: this.type = this

}


}

/**/
object TrainingHelp extends TrainingHelp_Scope0.TrainingHelp
              /*
                  -- GENERATED --
                  DATE: Fri May 18 18:41:46 PDT 2018
                  SOURCE: C:/develop/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingHelp.scala.html
                  HASH: c0692c7227e770b9e979ec918819f4f72ff5ccac
                  MATRIX: 596->1|729->39|757->41|880->138|892->142|942->172|2594->1797|2607->1801|2658->1831|3222->2367|3236->2371|3291->2404|3441->2526|3455->2530|3507->2560|3662->2687|3676->2691|3729->2722|3907->2872|3921->2876|3977->2910|4202->3107|4216->3111|4271->3144
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|65->41|65->41|65->41|78->54|78->54|78->54|79->55|79->55|79->55|80->56|80->56|80->56|81->57|81->57|81->57|83->59|83->59|83->59
                  -- GENERATED --
              */
          