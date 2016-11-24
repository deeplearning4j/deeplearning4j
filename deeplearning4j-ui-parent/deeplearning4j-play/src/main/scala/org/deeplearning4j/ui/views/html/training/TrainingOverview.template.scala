
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
        <title>"""),_display_(/*7.17*/i18n/*7.21*/.getMessage("train.pagetitle")),format.raw/*7.51*/("""</title>
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
                    <a class="brand" href="#"><span>"""),_display_(/*41.54*/i18n/*41.58*/.getMessage("train.pagetitle")),format.raw/*41.88*/("""</span></a>
                    <div id="sessionSelectDiv" style="display:none; float:right">
                        """),_display_(/*43.26*/i18n/*43.30*/.getMessage("train.session.label")),format.raw/*43.64*/("""
                        """),format.raw/*44.25*/("""<select id="sessionSelect" onchange='selectNewSession()'>
                            <option>(Session ID)</option>
                        </select>
                    </div>
                    <div id="workerSelectDiv" style="display:none; float:right;">
                        """),_display_(/*49.26*/i18n/*49.30*/.getMessage("train.session.worker.label")),format.raw/*49.71*/("""
                        """),format.raw/*50.25*/("""<select id="workerSelect" onchange='selectNewWorker()'>
                            <option>(Worker ID)</option>
                        </select>
                    </div>
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
                            <li class="active"><a href="javascript:void(0);"><i class="icon-bar-chart"></i><span class="hidden-tablet">"""),_display_(/*66.137*/i18n/*66.141*/.getMessage("train.nav.overview")),format.raw/*66.174*/("""</span></a></li>
                            <li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet">"""),_display_(/*67.104*/i18n/*67.108*/.getMessage("train.nav.model")),format.raw/*67.138*/("""</span></a></li>
                            <li><a href="system"><i class="icon-dashboard"></i><span class="hidden-tablet">"""),_display_(/*68.109*/i18n/*68.113*/.getMessage("train.nav.system")),format.raw/*68.144*/("""</span></a></li>
                            """),format.raw/*69.160*/("""
                            """),format.raw/*70.29*/("""<li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet">"""),_display_(/*71.146*/i18n/*71.150*/.getMessage("train.nav.language")),format.raw/*71.183*/("""</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        English</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ja', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        日本語</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('zh', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        中文</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ko', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        한글</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ru', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        русский</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('uk', 'overview')"><i class="icon-file-alt"></i><span class="hidden-tablet">
                                        український</span></a></li>
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

                    <!-- Start Score Chart-->
                <div id="content" class="span10">

                    <div class="row-fluid">

                        <div class="box span8">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*107.41*/i18n/*107.45*/.getMessage("train.overview.chart.scoreTitle")),format.raw/*107.91*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <div id="scoreiterchart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*111.55*/i18n/*111.59*/.getMessage("train.overview.chart.scoreTitleShort")),format.raw/*111.110*/("""
                                    """),format.raw/*112.37*/(""":</b> <span id="y">0</span>, <b>"""),_display_(/*112.70*/i18n/*112.74*/.getMessage("train.overview.charts.iteration")),format.raw/*112.120*/("""
                                    """),format.raw/*113.37*/(""":</b> <span id="x">
                                    0</span></p>
                            </div>
                        </div>
                            <!-- End Score Chart-->
                            <!-- Start Model Table-->
                        <div class="box span4">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*121.41*/i18n/*121.45*/.getMessage("train.overview.perftable.title")),format.raw/*121.90*/("""</b></h2>
                            </div>
                            <div class="box-content">
                                <table class="table table-bordered table-striped table-condensed">
                                    <tr>
                                        <td>"""),_display_(/*126.46*/i18n/*126.50*/.getMessage("train.overview.modeltable.modeltype")),format.raw/*126.100*/("""</td>
                                        <td id="modelType">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*130.46*/i18n/*130.50*/.getMessage("train.overview.modeltable.nLayers")),format.raw/*130.98*/("""</td>
                                        <td id="nLayers">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*134.46*/i18n/*134.50*/.getMessage("train.overview.modeltable.nParams")),format.raw/*134.98*/("""</td>
                                        <td id="nParams">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*138.46*/i18n/*138.50*/.getMessage("train.overview.perftable.startTime")),format.raw/*138.99*/("""</td>
                                        <td id="startTime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*142.46*/i18n/*142.50*/.getMessage("train.overview.perftable.totalRuntime")),format.raw/*142.102*/("""</td>
                                        <td id="totalRuntime">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*146.46*/i18n/*146.50*/.getMessage("train.overview.perftable.lastUpdate")),format.raw/*146.100*/("""</td>
                                        <td id="lastUpdate">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*150.46*/i18n/*150.50*/.getMessage("train.overview.perftable.totalParamUpdates")),format.raw/*150.107*/("""</td>
                                        <td id="totalParamUpdates">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*154.46*/i18n/*154.50*/.getMessage("train.overview.perftable.updatesPerSec")),format.raw/*154.103*/("""</td>
                                        <td id="updatesPerSec">Loading...</td>
                                    </tr>
                                    <tr>
                                        <td>"""),_display_(/*158.46*/i18n/*158.50*/.getMessage("train.overview.perftable.examplesPerSec")),format.raw/*158.104*/("""</td>
                                        <td id="examplesPerSec">Loading...</td>
                                    </tr>
                                </table>
                            </div>
                        </div>
                            <!--End Model Table -->
                    </div>


                    <div class="row-fluid">
                            <!--Start Ratio Table -->
                        <div class="box span6">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*172.41*/i18n/*172.45*/.getMessage("train.overview.chart.updateRatioTitle")),format.raw/*172.97*/(""": log<sub>10</sub></b></h2>
                            </div>
                            <div class="box-content">
                                <div id="updateRatioChart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*176.55*/i18n/*176.59*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*176.116*/("""
                                    """),format.raw/*177.37*/(""":</b> <span id="yRatio">0</span>, <b>log<sub>
                                    10</sub> """),_display_(/*178.47*/i18n/*178.51*/.getMessage("train.overview.chart.updateRatioTitleShort")),format.raw/*178.108*/("""
                                    """),format.raw/*179.37*/(""":</b> <span id="yLogRatio">0</span>
                                    , <b>"""),_display_(/*180.43*/i18n/*180.47*/.getMessage("train.overview.charts.iteration")),format.raw/*180.93*/(""":</b> <span id="xRatio">
                                        0</span></p>
                            </div>
                        </div>
                            <!--End Ratio Table -->
                            <!--Start Variance Table -->
                        <div class="box span6">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*188.41*/i18n/*188.45*/.getMessage("train.overview.chart.stdevTitle")),format.raw/*188.91*/(""": log<sub>10</sub></b></h2>
                                <ul class="nav tab-menu nav-tabs" style="position:absolute; margin-top: -11px; right: 22px;">
                                    <li class="active" id="stdevActivations"><a href="javascript:void(0);" onclick="selectStdevChart('stdevActivations')">"""),_display_(/*190.156*/i18n/*190.160*/.getMessage("train.overview.chart.stdevBtn.activations")),format.raw/*190.216*/("""</a></li>
                                    <li id="stdevGradients"><a href="javascript:void(0);" onclick="selectStdevChart('stdevGradients')">"""),_display_(/*191.137*/i18n/*191.141*/.getMessage("train.overview.chart.stdevBtn.gradients")),format.raw/*191.195*/("""</a></li>
                                    <li id="stdevUpdates"><a href="javascript:void(0);" onclick="selectStdevChart('stdevUpdates')">"""),_display_(/*192.133*/i18n/*192.137*/.getMessage("train.overview.chart.stdevBtn.updates")),format.raw/*192.189*/("""</a></li>
                                </ul>
                            </div>
                            <div class="box-content">
                                <div id="stdevChart" class="center" style="height: 300px;" ></div>
                                <p id="hoverdata"><b>"""),_display_(/*197.55*/i18n/*197.59*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*197.110*/("""
                                    """),format.raw/*198.37*/(""":</b> <span id="yStdev">0</span>, <b>log<sub>
                                    10</sub> """),_display_(/*199.47*/i18n/*199.51*/.getMessage("train.overview.chart.stdevTitleShort")),format.raw/*199.102*/("""
                                    """),format.raw/*200.37*/(""":</b> <span id="yLogStdev">0</span>
                                    , <b>"""),_display_(/*201.43*/i18n/*201.47*/.getMessage("train.overview.charts.iteration")),format.raw/*201.93*/(""":</b> <span id="xStdev">
                                        0</span></p>
                            </div>
                        </div>
                            <!-- End Variance Table -->
                    </div>
                </div>
            </div><!-- End Content Span10-->
        </div><!--End Row Fluid-->

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
        <script src="/assets/js/jquery.flot.selection.js"></script>
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
        <script src="/assets/js/train/overview.js"></script>    <!-- Charts and tables are generated here! -->
        <script src="/assets/js/train/train.js"></script>   <!-- Common (lang selection, etc) -->

        <!-- Execute once on page load -->
        <script>
                $(document).ready(function () """),format.raw/*247.47*/("""{"""),format.raw/*247.48*/("""
                    """),format.raw/*248.21*/("""renderOverviewPage(true);
                """),format.raw/*249.17*/("""}"""),format.raw/*249.18*/(""");
        </script>

            <!-- Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*254.41*/("""{"""),format.raw/*254.42*/("""
                    """),format.raw/*255.21*/("""renderOverviewPage(false);
                """),format.raw/*256.17*/("""}"""),format.raw/*256.18*/(""", 2000);
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
                  DATE: Tue Nov 22 21:45:11 PST 2016
                  SOURCE: /Users/justin/Projects/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingOverview.scala.html
                  HASH: f85d6d5e359ba9f03d31813d4bd0244a8c1633dd
                  MATRIX: 604->1|737->39|764->40|882->132|894->136|944->166|2686->1881|2699->1885|2750->1915|2896->2034|2909->2038|2964->2072|3017->2097|3328->2381|3341->2385|3403->2426|3456->2451|4194->3161|4208->3165|4263->3198|4411->3318|4425->3322|4477->3352|4630->3477|4644->3481|4697->3512|4771->3688|4828->3717|5006->3867|5020->3871|5075->3904|7517->6318|7531->6322|7599->6368|7883->6624|7897->6628|7971->6679|8037->6716|8098->6749|8112->6753|8181->6799|8247->6836|8657->7218|8671->7222|8738->7267|9050->7551|9064->7555|9137->7605|9374->7814|9388->7818|9458->7866|9693->8073|9707->8077|9777->8125|10012->8332|10026->8336|10097->8385|10334->8594|10348->8598|10423->8650|10663->8862|10677->8866|10750->8916|10988->9126|11002->9130|11082->9187|11327->9404|11341->9408|11417->9461|11658->9674|11672->9678|11749->9732|12332->10287|12346->10291|12420->10343|12724->10619|12738->10623|12818->10680|12884->10717|13004->10809|13018->10813|13098->10870|13164->10907|13270->10985|13284->10989|13352->11035|13774->11429|13788->11433|13856->11479|14194->11788|14209->11792|14288->11848|14463->11994|14478->11998|14555->12052|14726->12194|14741->12198|14816->12250|15134->12540|15148->12544|15222->12595|15288->12632|15408->12724|15422->12728|15496->12779|15562->12816|15668->12894|15682->12898|15750->12944|18374->15539|18404->15540|18454->15561|18525->15603|18555->15604|18719->15739|18749->15740|18799->15761|18871->15804|18901->15805
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|65->41|65->41|65->41|67->43|67->43|67->43|68->44|73->49|73->49|73->49|74->50|90->66|90->66|90->66|91->67|91->67|91->67|92->68|92->68|92->68|93->69|94->70|95->71|95->71|95->71|131->107|131->107|131->107|135->111|135->111|135->111|136->112|136->112|136->112|136->112|137->113|145->121|145->121|145->121|150->126|150->126|150->126|154->130|154->130|154->130|158->134|158->134|158->134|162->138|162->138|162->138|166->142|166->142|166->142|170->146|170->146|170->146|174->150|174->150|174->150|178->154|178->154|178->154|182->158|182->158|182->158|196->172|196->172|196->172|200->176|200->176|200->176|201->177|202->178|202->178|202->178|203->179|204->180|204->180|204->180|212->188|212->188|212->188|214->190|214->190|214->190|215->191|215->191|215->191|216->192|216->192|216->192|221->197|221->197|221->197|222->198|223->199|223->199|223->199|224->200|225->201|225->201|225->201|271->247|271->247|272->248|273->249|273->249|278->254|278->254|279->255|280->256|280->256
                  -- GENERATED --
              */
          