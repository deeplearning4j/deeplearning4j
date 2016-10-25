
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
		<link id="ie-style" href="/assets/css/ie.css" rel="stylesheet">
	<![endif]-->

	<!--[if IE 9]>
		<link id="ie9style" href="/assets/css/ie9.css" rel="stylesheet">
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

			<!-- Start Content -->
			<div id="content" class="span10">

			<div class="row-fluid">

				<div class="box blue span12">
					<div class="box-header">
						<h2>Network View</h2>
					</div>
					<div class="box-content">

						<a class="quick-button span2">
							<i class="glyphicons-icon database_plus"></i>
							<p>Input</p>
						</a>
						<a class="quick-button span2">
							<i class="glyphicons-icon picture"></i>
							<p>Convolution</p>
						</a>
						<a class="quick-button span2">
							<i class="glyphicons-icon cogwheels"></i>
							<p>Subsampling</p>
						</a>
						<a class="quick-button span2">
							<i class="glyphicons-icon picture"></i>
							<p>Convolution</p>
						</a>
						<a class="quick-button span2">
							<i class="glyphicons-icon cogwheels"></i>
							<p>Dense</p>
						</a>
						<a class="quick-button span2">
							<i class="glyphicons-icon check"></i>
							<p>Output</p>
						</a>
						<div class="clearfix"></div>
					</div>
				</div><!--/span-->

			</div><!--/row-->

			<div class="row-fluid sortable">
				<div class="box span12">
					<div class="box-header">
						<h2><b>Layer Information</b></h2>
					</div>
					<div class="box-content">
						<table class="table table-bordered table-striped table-condensed">
							  <thead>
								  <tr>
									  <th>Name</th>
									  <th>Type</th>
									  <th>Inputs</th>
									  <th>Outputs</th>
									  <th>Activation Function</th>
									  <th>Learning Rate</th>
								  </tr>
							  </thead>
							  <tbody>
								<tr>
									<td>Input</td>
									<td>Dense</td>
									<td>800</td>
									<td>500</td>
									<td>relu</td>
									<td>0.01</td>
								</tr>
							  </tbody>
						 </table>
					</div>
				</div><!--/span-->
			</div><!--/row-->

			<div class="box">
				<div class="box-header">
					<h2><b>Mean Magnitudes</b></h2>
				</div>
				<div class="box-content">
					<div id="sincos"  class="center" style="height:300px;" ></div>
					<p id="hoverdata"><b>Y:</b> <span id="y">0</span>, <b>X:</b> <span id="x">0</span></p>
				</div>
			</div>

			<div class="box">
				<div class="box-header">
					<h2><b>Activations</b></h2>
				</div>
				<div class="box-content">
					<div id="sincos2"  class="center" style="height:300px;" ></div>
					<p id="hoverdata"><b>Y:</b> <span id="y">0</span>, <b>X:</b> <span id="x">0</span></p>
				</div>
			</div>

			<div class="box">
				<div class="box-header">
					<h2><b>Learning Rate</b></h2>
				</div>
				<div class="box-content">
					<div id="sincos3"  class="center" style="height:300px;" ></div>
					<p id="hoverdata"><b>Y:</b> <span id="y">0</span>, <b>X:</b> <span id="x">0</span></p>
				</div>
			</div>

			<div class="box">
				<div class="box-header">
					<h2><b>Parameters vs Updates</b></h2>
				</div>
				<div class="box-content">
					 <div id="stackchart" class="center" style="height:300px;"></div>

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
<<<<<<< HEAD
                  DATE: Sun Oct 23 22:17:19 PDT 2016
                  SOURCE: /Users/ejunprung/skymind-ui/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingModel.scala.html
                  HASH: e714fa49c4538990e1e34df71c54ef8f26d13eaa
                  MATRIX: 598->1|731->39|758->40
=======
                  DATE: Tue Oct 25 20:32:36 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingModel.scala.html
                  HASH: ed881b56c2e3c64aa44d53af9c02b046f33f6e3f
                  MATRIX: 598->1|731->39|759->41
>>>>>>> ab_ui
                  LINES: 20->1|25->1|26->2
                  -- GENERATED --
              */
          