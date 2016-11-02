
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingModel_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingModel extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply/*1.2*/(i18n: org.deeplearning4j.ui.api.I18N):play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.40*/("""
"""),format.raw/*2.1*/("""<!DOCTYPE html>
<html lang="en">
<head>

	<meta charset="utf-8">
	<title>System Page Demo</title>
	<!-- start: Mobile Specific -->
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<!-- end: Mobile Specific -->

	<link id="bootstrap-style" href="/assets/css/bootstrap.min.css" rel="stylesheet">
	<link href="/assets/css/bootstrap-responsive.min.css" rel="stylesheet">
	<link id="base-style" href="/assets/css/style.css" rel="stylesheet">
	<link id="base-style-responsive" href="/assets/css/style-responsive.css" rel="stylesheet">
	<link href='http://fonts.googleapis.com/css?family=Open+Sans:300italic,400italic,600italic,700italic,800italic,400,300,600,700,800&subset=latin,cyrillic-ext,latin-ext' rel='stylesheet' type='text/css'>
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
				<a class="brand" href="index.html"><span>DL4J Training UI</span></a>
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
						<li><a href="overview"><i class="icon-bar-chart"></i><span class="hidden-tablet"> Overview</span></a></li>
						<li class="active"><a href="javascript:void(0);"><i class="icon-tasks"></i><span class="hidden-tablet"> Model</span></a></li>
						<li><a href="system"><i class="icon-dashboard"></i><span class="hidden-tablet"> System</span></a></li>
						<li><a href="help"><i class="icon-star"></i><span class="hidden-tablet"> User Guide</span></a></li>
						<li>
							<a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet"> Language</span></a>
							<ul>
								<li><a class="submenu" href="javascript:void(0);"><i class="icon-file-alt"></i><span class="hidden-tablet"> English</span></a></li>
								<li><a class="submenu" href="javascript:void(0);"><i class="icon-file-alt"></i><span class="hidden-tablet"> Japanese</span></a></li>
								<li><a class="submenu" href="javascript:void(0);"><i class="icon-file-alt"></i><span class="hidden-tablet"> Chinese</span></a></li>
								<li><a class="submenu" href="javascript:void(0);"><i class="icon-file-alt"></i><span class="hidden-tablet"> Korean</span></a></li>
							</ul>
						</li>
					</ul>
				</div>
			</div>
			<!-- End Main Menu -->

			<noscript>
				<div class="alert alert-block span10">
					<h4 class="alert-heading">Warning!</h4>
					<p>You need to have <a href="http://en.wikipedia.org/wiki/JavaScript" target="_blank">JavaScript</a> enabled to use this site.</p>
				</div>
			</noscript>

			<style>
			/* Graph */
			#layers """),format.raw/*80.12*/("""{"""),format.raw/*80.13*/("""
			  """),format.raw/*81.6*/("""height: 100%;
			  width: 50%;
			  position: absolute;
			  left: 0;
			  top: 0;
			"""),format.raw/*86.4*/("""}"""),format.raw/*86.5*/("""
			"""),format.raw/*87.4*/("""</style>

			<!-- Start Content -->
			<div id="content" class="span10">

				<div class="row-fluid span6">
					<div id="layers"></div>
				</div>
				<!-- Start Layer Details -->
				<div class="row-fluid span6" id="0">

					<div class="box">
						<div class="box-header">
							<h2><b>Layer Information</b></h2>
						</div>
						<div class="box-content">
							<table class="table table-bordered table-striped table-condensed">
								<thead>
								<tr>
									<th>Name</th>
									<th>Type</th>
									<th>Input Size</th>
									<th>Output Size</th>
									<th># Parameters</th>
									<th>Activation Function</th>
									<th>Loss Function</th>
								</tr>
								</thead>
								<tbody>
								<tr>
									<td id="layerName">Loading...</td>
									<td id="layerType">Loading...</td>
									<td id="inputSize">Loading...</td>
									<td id="outputSize">Loading...</td>
									<td id="nParams">Loading...</td>
									<td id="activationFunction">Loading...</td>
									<td id="lossFunction">Loading...</td>
								</tr>
								</tbody>
							</table>
						</div>
					</div>

					<div class="box">
						<div class="box-header">
							<h2><b>Mean Magnitudes</b></h2>
						</div>
						<div class="box-content">
							<div id="meanmag"  class="center" style="height:300px;" ></div>
							<p id="hoverdata"><b>Magnitude:</b> <span id="y">0</span>, <b>Iteration:</b> <span id="x">0</span></p>
						</div>
					</div>

					<div class="box">
						<div class="box-header">
							<h2><b>Activations</b></h2>
						</div>
						<div class="box-content">
							<div id="activations"  class="center" style="height:300px;" ></div>
							<p id="hoverdata"><b>Activation:</b> <span id="y">0</span>, <b>Iteration:</b> <span id="x">0</span></p>
						</div>
					</div>

					<div class="box">
						<div class="box-header">
							<h2><b>Learning Rates</b></h2>
						</div>
						<div class="box-content">
							<div id="learningrate"  class="center" style="height:300px;" ></div>
							<p id="hoverdata"><b>Learning Rate:</b> <span id="y">0</span>, <b>Iteration:</b> <span id="x">0</span></p>
						</div>
					</div>

					<div class="box">
						<div class="box-header">
							<h2><b>Parameters Histogram</b></h2>
						</div>
						<div class="box-content">
							<div id="parametershistogram" class="center" style="height:300px;"></div>

							<p class="stackControls center">
								<input class="btn" type="button" value="With stacking">
								<input class="btn" type="button" value="Without stacking">
							</p>

							<p class="graphControls center">
								<input class="btn-primary" type="button" value="Bars">
								<input class="btn-primary" type="button" value="Lines">
								<input class="btn-primary" type="button" value="Lines with steps">
							</p>
						</div>
					</div>

				</div>
				<!-- End Layer Details-->

		<!-- End Content -->
		</div><!-- End Container -->
	</div><!-- End Row Fluid-->

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
		<script src="/assets/js/custom.js"></script>
		<script src="/assets/js/cytoscape.min.js"></script>
		<script src="/assets/js/dagre.min.js"></script>
		<script src="/assets/js/cytoscape-dagre.js"></script>
		"""),format.raw/*221.100*/("""
		"""),format.raw/*222.3*/("""<script src="/assets/js/train/model.js"></script>        <!-- Charts and tables are generated here! -->

        <script src="/assets/js/train/model-graph.js"></script> <!-- Layer graph generated here! -->

		<!-- Execute once on page load -->
		<script>
				$(document).ready(function()"""),format.raw/*228.33*/("""{"""),format.raw/*228.34*/("""
					"""),format.raw/*229.6*/("""renderModelGraph();
					renderLayerTable();
					renderModelPage();
				"""),format.raw/*232.5*/("""}"""),format.raw/*232.6*/(""");
		</script>

		<!-- Execute periodically (every 2 sec) -->
		<script>
				setInterval(function()"""),format.raw/*237.27*/("""{"""),format.raw/*237.28*/("""
					"""),format.raw/*238.6*/("""renderLayerTable();
					renderModelPage();
				"""),format.raw/*240.5*/("""}"""),format.raw/*240.6*/(""", 2000);
		</script>

		<!--<script type="text/javascript">-->
		<!--$(document).ready(function() """),format.raw/*244.36*/("""{"""),format.raw/*244.37*/("""-->
			<!--var option = '1';-->
			<!--var url = window.location.href;-->
			<!--option = url.match(/layer=(.*)/)[1];-->
			<!--showDiv(option);-->
		<!--"""),format.raw/*249.7*/("""}"""),format.raw/*249.8*/(""");-->
		<!--function showDiv(option) """),format.raw/*250.32*/("""{"""),format.raw/*250.33*/("""-->
			<!--$('#0').hide();-->
			<!--$('#' + option).show();-->
		<!--"""),format.raw/*253.7*/("""}"""),format.raw/*253.8*/("""-->
		<!--</script>-->
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
object TrainingModel extends TrainingModel_Scope0.TrainingModel
              /*
                  -- GENERATED --
                  DATE: Wed Nov 02 17:56:03 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingModel.scala.html
                  HASH: c21010b2d872f69e486dededfddda2d2236ebb41
                  MATRIX: 598->1|731->39|759->41|4296->3550|4325->3551|4359->3558|4477->3649|4505->3650|4537->3655|9593->8778|9625->8782|9947->9075|9977->9076|10012->9083|10116->9159|10145->9160|10278->9264|10308->9265|10343->9272|10421->9322|10450->9323|10581->9425|10611->9426|10798->9585|10827->9586|10894->9624|10924->9625|11025->9698|11054->9699
                  LINES: 20->1|25->1|26->2|104->80|104->80|105->81|110->86|110->86|111->87|245->221|246->222|252->228|252->228|253->229|256->232|256->232|261->237|261->237|262->238|264->240|264->240|268->244|268->244|273->249|273->249|274->250|274->250|277->253|277->253
                  -- GENERATED --
              */
          