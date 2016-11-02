
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
						<li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet"> Model</span></a></li>
						<li class="active"><a href="javascript:void(0);"><i class="icon-dashboard"></i><span class="hidden-tablet"> System</span></a></li>
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

			<!-- Start Content -->
			<div id="content" class="span10">

				<div class="row-fluid">

					<div class="box span12">
						<div class="box-header">
							<h2><b>Select Machine</b></h2>
						</div>
						<div class="box-content">
							<ul class="nav tab-menu nav-tabs" id="systemTab"></ul>

							<!--Start System Tab -->
							<div id="myTabContent" class="tab-content">
								<div class="tab-pane active" id="machine1">

									<!-- JVM Memory Utilization Chart -->
									<div class="row-fluid">

                                        <div class="box span6">
                                            <div class="box-header">
                                                <h2><b>"""),_display_(/*100.57*/i18n/*100.61*/.getMessage("train.system.chart.jvmTitle")),format.raw/*100.103*/("""</b></h2>
                                            </div>
                                            <div class="box-content">
                                                <div id="jvmmemorychart" class="center" style="height: 300px;" ></div>
                                                <p id="hoverdata"><b>JVM Memory:</b> <span id="y">0</span>, <b>Iteration:</b> <span id="x">0</span></p>
                                            </div>
										</div>
                                        <!-- Off Heap Memory Utlization Chart -->
                                        <div class="box span6">
                                            <div class="box-header">
                                                <h2><b>"""),_display_(/*110.57*/i18n/*110.61*/.getMessage("train.system.chart.offHeapTitle")),format.raw/*110.107*/("""</b></h2>
                                            </div>
                                            <div class="box-content">
                                                <div id="offheapmemorychart" class="center" style="height: 300px;" ></div>
                                                <p id="hoverdata"><b>Off Heap Memory:</b> <span id="y2">0</span>, <b>Iteration:</b> <span id="x2">0</span></p>
                                            </div>
                                        </div>

									</div>

									<!-- Tables -->
                                    <div class="row-fluid">

                                        <!-- Hardware Information -->
                                        <div class="box span12">
                                            <div class="box-header">
                                                <h2><b>"""),_display_(/*126.57*/i18n/*126.61*/.getMessage("train.system.table.hardwareTitle")),format.raw/*126.108*/("""</b></h2>
                                            </div>
                                            <div class="box-content">
                                                <table class="table table-striped">
                                                    <thead>
                                                    <tr>
                                                        <th>JVM Current Memory</th>
                                                        <th>JVM Max Memory</th>
                                                        <th>Off-Heap Current Memory</th>
                                                        <th>Off-Heap Max Memory</th>
                                                        <th>JVM Available Processors</th>
                                                        <th># Compute Devices</th>
                                                    </tr>
                                                    </thead>
                                                    <tbody>
                                                    <tr>
                                                        <td id="currentBytesJVM">Loading...</td>
                                                        <td id="maxBytesJVM">Loading...</td>
                                                        <td id="currentBytesOffHeap">Loading...</td>
                                                        <td id="maxBytesOffHeap">Loading...</td>
                                                        <td id="jvmAvailableProcessors">Loading...</td>
                                                        <td id="nComputeDevices">Loading...</td>
                                                    </tr>
                                                    </tbody>
                                                </table>
                                            </div>
                                        </div>

                                    </div>

									<div class="row-fluid">

										<!-- Software Information -->
										<div class="box span12">
											<div class="box-header">
												<h2><b>"""),_display_(/*161.21*/i18n/*161.25*/.getMessage("train.system.table.softwareTitle")),format.raw/*161.72*/("""</b></h2>
											</div>
											<div class="box-content">
												<table class="table table-striped">
													<thead>
													<tr>
														<th>OS</th>
														<th>Host Name</th>
														<th>OS Architecture</th>
														<th>JVM Name</th>
														<th>JVM Version</th>
														<th>ND4J Backend</th>
														<th>ND4J Data Type</th>
													</tr>
													</thead>
													<tbody>
													<tr>
														<td id="OS">Loading...</td>
														<td id="hostName">Loading...</td>
														<td id="OSArchitecture">Loading...</td>
														<td id="jvmName">Loading...</td>
														<td id="jvmVersion">Loading...</td>
														<td id="nd4jBackend">Loading...</td>
														<td id="nd4jDataType">Loading...</td>
													</tr>
													</tbody>
												</table>
											</div>
										</div>

									</div>

									<!-- GPU Chart -->
									<div class="row-fluid">
										<div class="box span12">
											<div class="box-header">
												<h2><b>GPU Information (if isDevice == true)</b></h2>
											</div>
											<div class="box-content">
												<table class="table table-striped">
													<thead>
													<tr>
														<th>?</th>
														<th>?</th>
														<th>?</th>
														<th>?</th>
														<th>?</th>
														<th>?</th>
														<th>?</th>
													</tr>
													</thead>
													<tbody>
													<tr>
														<td id="gpuPlaceholder">Loading...</td>
														<td id="gpuPlaceholder">Loading...</td>
														<td id="gpuPlaceholder">Loading...</td>
														<td id="gpuPlaceholder">Loading...</td>
														<td id="gpuPlaceholder">Loading...</td>
														<td id="gpuPlaceholder">Loading...</td>
														<td id="gpuPlaceholder">Loading...</td>
													</tr>
													</tbody>
												</table>
											</div>
										</div>
									</div>
								</div>
								<!-- End System Tab -->
								<div class="tab-pane" id="machine2">
									<p>
										Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.
									</p>
								</div>

								<div class="tab-pane" id="machine3">
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
		<script src="/assets/js/train/system.js"></script> <!-- Charts and tables are generated here! -->

		<!-- Execute once on page load -->
		<script>
            $(document).ready(function()"""),format.raw/*283.41*/("""{"""),format.raw/*283.42*/("""
				"""),format.raw/*284.5*/("""renderSystemPage();
				renderMultipleTabs();
            """),format.raw/*286.13*/("""}"""),format.raw/*286.14*/(""");
        </script>

		 <!--Execute periodically (every 2 sec) -->
		<script>
            setInterval(function()"""),format.raw/*291.35*/("""{"""),format.raw/*291.36*/("""
                """),format.raw/*292.17*/("""renderSystemPage();
            """),format.raw/*293.13*/("""}"""),format.raw/*293.14*/(""", 2000);
        </script>
		 <!--End JavaScript-->

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
                  DATE: Wed Nov 02 20:14:42 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingSystem.scala.html
                  HASH: cc450ae1585c1c4636b4266ed9b29c48782ed59a
                  MATRIX: 600->1|733->39|761->41|5180->4432|5194->4436|5259->4478|6038->5229|6052->5233|6121->5279|7040->6170|7054->6174|7124->6221|9339->8408|9353->8412|9422->8459|15196->14204|15226->14205|15260->14211|15349->14271|15379->14272|15526->14390|15556->14391|15603->14409|15665->14442|15695->14443
                  LINES: 20->1|25->1|26->2|124->100|124->100|124->100|134->110|134->110|134->110|150->126|150->126|150->126|185->161|185->161|185->161|307->283|307->283|308->284|310->286|310->286|315->291|315->291|316->292|317->293|317->293
                  -- GENERATED --
              */
          