
package org.deeplearning4j.ui.views.html.training

import play.twirl.api._
import play.twirl.api.TemplateMagic._


     object TrainingOverview_Scope0 {
import models._
import controllers._
import play.api.i18n._
import views.html._
import play.api.templates.PlayMagic._
import play.api.mvc._
import play.api.data._

class TrainingOverview extends BaseScalaTemplate[play.twirl.api.HtmlFormat.Appendable,Format[play.twirl.api.HtmlFormat.Appendable]](play.twirl.api.HtmlFormat) with play.twirl.api.Template1[org.deeplearning4j.ui.api.I18N,play.twirl.api.HtmlFormat.Appendable] {

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
                    <a class="brand" href="#"><span>DL4J Training UI</span></a>
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
                            <li class="active"><a href="javascript:void(0);"><i class="icon-bar-chart"></i><span class="hidden-tablet">
                                Overview</span></a></li>
                            <li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet">
                                Model</span></a></li>
                            <li><a href="system"><i class="icon-dashboard"></i><span class="hidden-tablet">
                                System</span></a></li>
                            <li><a href="help"><i class="icon-star"></i><span class="hidden-tablet">
                                User Guide</span></a></li>
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

                    <div class="row-fluid">

                        <div class="box">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*82.41*/i18n/*82.45*/.getMessage("train.overview.chart.scoreTitle")),format.raw/*82.91*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <div id="scoreiterchart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>Score:</b> <span id="y">0</span>, <b>Iteration:</b> <span id="x">
                                    0</span></p>
                            </div>
                        </div>

                        <div class="row-fluid">
                            <div class="box span12">
                                <div class="box-header">
                                    <h2><b>Score vs Iteration: Realtime</b></h2>
                                </div>
                                <div class="box-content">
                                    <div id="realtimechart" style="height: 190px;"></div><br>
                                        <!--<p>Time between updates: <input id="updateInterval" type="text" value="" style="text-align: right; width:5em"> milliseconds</p>-->
                                </div>
                            </div>
                        </div><!--/row-->

                    </div>

                    <div class="row-fluid">
                        <div class="box span12">
                            <div class="box-header">
                                <h2><b>Model Performance</b></h2>
                            </div>
                            <div class="box-content">
                                <table class="table table-bordered table-striped table-condensed">
                                    <thead>
                                        <tr>
                                            <th>Model</th>
                                            <th>Score</th>
                                            <th>Iteration</th>
                                            <th>Status</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <tr>
                                            <td>Model 1</td>
                                            <td class="center">123141</td>
                                            <td class="center">123213213</td>
                                            <td class="center">
                                                <span class="label label-success">Completed</span>
                                            </td>
                                        </tr>
                                        <tr>
                                            <td>Model 2</td>
                                            <td class="center">12321321</td>
                                            <td class="center">123213</td>
                                            <td class="center">
                                                <span class="label label-important">Failed</span>
                                            </td>
                                        </tr>
                                        <tr>
                                            <td>Model 3</td>
                                            <td class="center">123213</td>
                                            <td class="center">123123</td>
                                            <td class="center">
                                                <span class="label label-warning">Pending</span>
                                            </td>
                                        </tr>
                                        <tr>
                                            <td>Model 4</td>
                                            <td class="center">12312321</td>
                                            <td class="center">12313112</td>
                                            <td class="center">
                                                <span class="label label-warning">Pending</span>
                                            </td>
                                        </tr>
                                        <tr>
                                            <td>Model 5</td>
                                            <td class="center">12313123213</td>
                                            <td class="center">123213211</td>
                                            <td class="center">
                                                <span class="label label-success">Completed</span>
                                            </td>
                                        </tr>
                                    </tbody>
                                </table>
                                <div class="pagination pagination-centered">
                                    <ul>
                                        <li><a href="#">Prev</a></li>
                                        <li class="active">
                                            <a href="#">1</a>
                                        </li>
                                        <li><a href="#">2</a></li>
                                        <li><a href="#">3</a></li>
                                        <li><a href="#">4</a></li>
                                        <li><a href="#">Next</a></li>
                                    </ul>
                                </div>
                            </div>
                        </div><!--/span-->
                    </div><!--/row-->

                    <!-- end: Content -->
                </div><!--/#content.span10-->
            </div><!--/fluid-row-->

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
            """),format.raw/*213.61*/("""
            """),format.raw/*214.13*/("""<script src="/assets/js/train/overview.js"></script>    <!-- Charts are generated here! -->

            <!-- Execute once on page load -->
            <script>
                    $(document).ready(function()"""),format.raw/*218.49*/("""{"""),format.raw/*218.50*/("""
                        """),format.raw/*219.25*/("""renderScoreChart();
                    """),format.raw/*220.21*/("""}"""),format.raw/*220.22*/(""");
            </script>

            <!-- Execute periodically -->
            <script>

            </script>
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
object TrainingOverview extends TrainingOverview_Scope0.TrainingOverview
              /*
                  -- GENERATED --
                  DATE: Tue Oct 25 23:25:12 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingOverview.scala.html
                  HASH: cf4bb3190c83b35283ce69b3af68303ed25e9346
                  MATRIX: 604->1|737->39|765->41|4580->3829|4593->3833|4660->3879|12507->11745|12550->11759|12792->11972|12822->11973|12877->11999|12947->12040|12977->12041
                  LINES: 20->1|25->1|26->2|106->82|106->82|106->82|237->213|238->214|242->218|242->218|243->219|244->220|244->220
                  -- GENERATED --
              */
          