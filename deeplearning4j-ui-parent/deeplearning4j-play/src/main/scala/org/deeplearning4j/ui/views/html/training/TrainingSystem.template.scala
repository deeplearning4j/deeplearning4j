
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingSystem_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingSystem extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

  /**/
  def apply/*1.2*/(i18n: org.deeplearning4j.ui.api.I18N):play.twirl.api.HtmlFormat.Appendable = {
    _display_ {
      {


Seq[Any](format.raw/*1.40*/("""
"""),format.raw/*2.1*/("""<!DOCTYPE html>
<html lang="en">
<head>

	<meta charset="utf-8">
	<title>DL4J Training UI</title>
	<!-- Start Mobile Specific -->
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<!-- End Mobile Specific -->

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
						<li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet"> Model</span></a></li>
						<li class="active"><a href="javascript:void(0);"><i class="icon-dashboard"></i><span class="hidden-tablet"> System</span></a></li>
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

				<div class="box span12">
					<div class="box-header">
						<h2><b>Select Machine</b></h2>
					</div>
					<div class="box-content">
						<ul class="nav tab-menu nav-tabs" id="myTab">
							<li class="active"><a href="#system1">System 1</a></li>
							<li><a href="#system2">System 2</a></li>
							<li><a href="#system3">System 3</a></li>
						</ul>

						<!--Start System Tab -->
						<div id="myTabContent" class="tab-content">
							<div class="tab-pane active" id="system1">

							<!-- Start Memory Utilization -->
							<div class="row-fluid">

								<div class="span8 widget blue" onTablet="span7" onDesktop="span8">
									<h1>Memory Utilization</h1>
									<div id="stats-chart2"  style="height:282px" ></div>
								</div>

								<!-- Start System Statistics -->
								<div class="sparkLineStats span4 widget green" onTablet="span5" onDesktop="span4">
									<h1>System Statistics</h1>
			              <ul class="unstyled">
		                  <li><span class="sparkLineStats3"></span>
		                      Memory:
		                      <span class="number">781</span>
		                  </li>
		                  <li><span class="sparkLineStats4"></span>
		                      Series Name:
		                      <span class="number">2,19</span>
		                  </li>
		                  <li><span class="sparkLineStats5"></span>
		                      Device:
		                      <span class="number">00:02:58</span>
		                  </li>
		                  <li><span class="sparkLineStats6"></span>
		                      Times: <span class="number">59,83%</span>
		                  </li>
		                  <li><span class="sparkLineStats7"></span>
		                      Values:
		                      <span class="number">70,79%</span>
		                  </li>
		                  <li><span class="sparkLineStats8"></span>
		                      Max Bytes:
		                      <span class="number">29,21%</span>
		                  </li>
			              </ul>
								</div>
							</div>
							<div class="row-fluid hideInIE8 circleStats">

								<div class="span2" onTablet="span4" onDesktop="span2">
				          <div class="circleStatsItemBox yellow">
										<div class="header">Disk Space Usage</div>
										<span class="percent">percent</span>
										<div class="circleStat"><input type="text" value="58" class="whiteCircle" /></div>
										<div class="footer">
											<span class="count">
												<span class="number">20000</span>
												<span class="unit">MB</span>
											</span>
											<span class="sep"> / </span>
											<span class="value">
												<span class="number">50000</span>
												<span class="unit">MB</span>
											</span>
										</div>
				          </div>
								</div>

								<div class="span2" onTablet="span4" onDesktop="span2">
				          <div class="circleStatsItemBox green">
										<div class="header">Bandwidth</div>
										<span class="percent">percent</span>
										<div class="circleStat"><input type="text" value="78" class="whiteCircle" /></div>
										<div class="footer">
											<span class="count">
												<span class="number">5000</span>
												<span class="unit">GB</span>
											</span>
											<span class="sep"> / </span>
											<span class="value">
												<span class="number">5000</span>
												<span class="unit">GB</span>
											</span>
										</div>
				        	</div>
								</div>

								<div class="span2" onTablet="span4" onDesktop="span2">
				          <div class="circleStatsItemBox greenDark">
										<div class="header">Memory</div>
										<span class="percent">percent</span>
				            <div class="circleStat"><input type="text" value="100" class="whiteCircle" /></div>
										<div class="footer">
											<span class="count">
												<span class="number">64</span>
												<span class="unit">GB</span>
											</span>
											<span class="sep"> / </span>
											<span class="value">
												<span class="number">64</span>
												<span class="unit">GB</span>
											</span>
										</div>
				          </div>
								</div>

								<div class="span2 noMargin" onTablet="span4" onDesktop="span2">
				          <div class="circleStatsItemBox pink">
										<div class="header">CPU</div>
										<span class="percent">percent</span>
				            <div class="circleStat"><input type="text" value="83" class="whiteCircle" /></div>
										<div class="footer">
											<span class="count">
												<span class="number">64</span>
												<span class="unit">GHz</span>
											</span>
											<span class="sep"> / </span>
											<span class="value">
												<span class="number">3.2</span>
												<span class="unit">GHz</span>
											</span>
										</div>
				          </div>
								</div>

								<div class="span2" onTablet="span4" onDesktop="span2">
				          <div class="circleStatsItemBox orange">
										<div class="header">Memory</div>
										<span class="percent">percent</span>
				            <div class="circleStat"><input type="text" value="100" class="whiteCircle" /></div>
										<div class="footer">
											<span class="count">
												<span class="number">64</span>
												<span class="unit">GB</span>
											</span>
											<span class="sep"> / </span>
											<span class="value">
												<span class="number">64</span>
												<span class="unit">GB</span>
											</span>
										</div>
				          </div>
								</div>

								<div class="span2" onTablet="span4" onDesktop="span2">
				          <div class="circleStatsItemBox greenLight">
										<div class="header">GPU</div>
										<span class="percent">percent</span>
				            <div class="circleStat"><input type="text" value="100" class="whiteCircle" /></div>
										<div class="footer">
											<span class="count">
												<span class="number">64</span>
												<span class="unit">GB</span>
											</span>
											<span class="sep"> / </span>
											<span class="value">
												<span class="number">64</span>
												<span class="unit">GB</span>
											</span>
										</div>
				          </div>
								</div>

							</div>

								<!-- Start Charts -->
								<div class="row-fluid">
								<!-- Start Software Information -->
									<div class="box span6">
										<div class="box-header">
											<h2><b>Hardware Information</b></h2>
										</div>
										<div class="box-content">
											<table class="table table-striped">
												  <thead>
													  <tr>
														  <th>HwJvmMaxMemory</th>
														  <th>HwOffHeapMaxMemory</th>
														  <th>HwJvmAvailableProcessor</th>
														  <th>numDevices</th>
														  <th>deviceName</th>
														  <th>deviceMemory</th>
													  </tr>
												  </thead>
												  <tbody>
													<tr>
														<td>12313123</td>
														<td>12313123</td>
														<td>12313123</td>
														<td>12313123</td>
														<td>device123</td>
														<td>12313123</td>
													</tr>
													<tr>
														<td>12313123</td>
														<td>12313123</td>
														<td>12313123</td>
														<td>12313123</td>
														<td>device123</td>
														<td>12313123</td>
													</tr>
													<tr>
														<td>12313123</td>
														<td>12313123</td>
														<td>12313123</td>
														<td>12313123</td>
														<td>device123</td>
														<td>12313123</td>
													</tr>
													<tr>
														<td>12313123</td>
														<td>12313123</td>
														<td>12313123</td>
														<td>12313123</td>
														<td>device123</td>
														<td>12313123</td>
													</tr>
													<tr>
														<td>12313123</td>
														<td>12313123</td>
														<td>12313123</td>
														<td>12313123</td>
														<td>device123</td>
														<td>12313123</td>
													</tr>
													<tr>
														<td>12313123</td>
														<td>12313123</td>
														<td>12313123</td>
														<td>12313123</td>
														<td>device123</td>
														<td>12313123</td>
													</tr>
												  </tbody>
											 </table>
										</div>
									</div>

									<!-- Start Software Information -->
									<div class="box span6">
										<div class="box-header">
											<h2><b>Software Information</b></h2>
										</div>
										<div class="box-content">
											<table class="table table-striped">
												  <thead>
													  <tr>
														  <th>OS</th>
														  <th>Hostname</th>
														  <th>Architecture</th>
														  <th>JVM Name</th>
														  <th>JVM Version</th>
														  <th>ND4J Backend</th>
														  <th>ND4J DataType</th>
													  </tr>
												  </thead>
												  <tbody>
													<tr>
														<td>OS123</td>
														<td class="center">Hostname 123</td>
														<td class="center">Architecture 123</td>
														<td class="center">JVM 123</td>
														<td class="center">JVM Version 123</td>
														<td class="center">ND4J Backend 123</td>
														<td class="center">ND4J Datatype 123</td>
													</tr>
													<tr>
														<td>OS123</td>
														<td class="center">Hostname 123</td>
														<td class="center">Architecture 123</td>
														<td class="center">JVM 123</td>
														<td class="center">JVM Version 123</td>
														<td class="center">ND4J Backend 123</td>
														<td class="center">ND4J Datatype 123</td>
													</tr>
													<tr>
														<td>OS123</td>
														<td class="center">Hostname 123</td>
														<td class="center">Architecture 123</td>
														<td class="center">JVM 123</td>
														<td class="center">JVM Version 123</td>
														<td class="center">ND4J Backend 123</td>
														<td class="center">ND4J Datatype 123</td>
													</tr>
													<tr>
														<td>OS123</td>
														<td class="center">Hostname 123</td>
														<td class="center">Architecture 123</td>
														<td class="center">JVM 123</td>
														<td class="center">JVM Version 123</td>
														<td class="center">ND4J Backend 123</td>
														<td class="center">ND4J Datatype 123</td>
													</tr>
												  </tbody>
											 </table>
										</div>
									</div>

								</div>
							<!--End System Tab -->
							</div>

							<div class="tab-pane" id="system2">
								<p>
									Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.
								</p>
							</div>

							<div class="tab-pane" id="system3">
								<p>
									Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat. Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi.
								</p>
							</div>
						</div>
					</div>
				</div>
				<!-- End System Tab -->
			</div><!-- End Row Fluid-->
			</div><!-- End Content -->
		</div><!-- End Container-->
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
object TrainingSystem extends TrainingSystem_Scope0.TrainingSystem
              /*
                  -- GENERATED --
                  DATE: Tue Oct 25 22:59:20 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingSystem.scala.html
                  HASH: f929a354ca257e8c1c287d1b0efd8713d8967eff
                  MATRIX: 600->1|733->39|761->41
                  LINES: 20->1|25->1|26->2
                  -- GENERATED --
              */
          