
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
                    <a class="brand" href="index.html"><span>"""),_display_(/*41.63*/i18n/*41.67*/.getMessage("train.pagetitle")),format.raw/*41.97*/("""</span></a>
                    <div id="sessionSelectDiv" style="display:none; float:right">
                        Session:
                        <select id="sessionSelect" onchange='selectNewSession()'>
                            <option>(Session ID)</option>
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
                            <li><a href="overview"><i class="icon-bar-chart"></i><span class="hidden-tablet"> """),_display_(/*60.112*/i18n/*60.116*/.getMessage("train.nav.overview")),format.raw/*60.149*/("""</span></a></li>
                            <li><a href="model"><i class="icon-tasks"></i><span class="hidden-tablet"> """),_display_(/*61.105*/i18n/*61.109*/.getMessage("train.nav.model")),format.raw/*61.139*/("""</span></a></li>
                            <li class="active"><a href="javascript:void(0);"><i class="icon-dashboard"></i><span class="hidden-tablet"> """),_display_(/*62.138*/i18n/*62.142*/.getMessage("train.nav.system")),format.raw/*62.173*/("""</span></a></li>
                            <li><a href="help"><i class="icon-star"></i><span class="hidden-tablet"> """),_display_(/*63.103*/i18n/*63.107*/.getMessage("train.nav.userguide")),format.raw/*63.141*/("""</span></a></li>
                            <li>
                                <a class="dropmenu" href="javascript:void(0);"><i class="icon-folder-close-alt"></i><span class="hidden-tablet">
                                    Language</span></a>
                                <ul>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> English</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ja', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 日本語</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('zh', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 中文</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ko', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> 한글</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ru', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> русский</span></a></li>
                                    <li><a class="submenu" href="javascript:void(0);" onclick="languageSelect('uk', 'system')"><i class="icon-file-alt"></i> <span class="hidden-tablet"> український</span></a></li>
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

                    <div class="row-fluid">

                        <div class="box span12">
                            <div class="box-header">
                                <h2><b>"""),_display_(/*96.41*/i18n/*96.45*/.getMessage("train.system.title")),format.raw/*96.78*/("""</b></h2>
                                <div class="btn-group" style="margin-top: -11px; position:absolute; right: 40px;">
                                <button class="btn dropdown-toggle btn-primary" data-toggle="dropdown">Select Machine <span class="caret"></span></button>
                                    <ul class="dropdown-menu" id="systemTab"></ul>
                                </div>
                            </div>
                            <div class="box-content">

                                    <!--Start System Information -->
                                <div class="tab-content">
                                    <div class="tab-pane active">

                                            <!-- System Memory Utilization Chart -->
                                        <div class="row-fluid">

                                            <div class="box span12" id="systemMemoryChart">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*113.61*/i18n/*113.65*/.getMessage("train.system.chart.systemTitle")),format.raw/*113.110*/(""" """),format.raw/*113.111*/("""%</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <div id="systemMemoryChartPlot" class="center" style="height: 300px;" ></div>
                                                    <p id="hoverdata"><b>"""),_display_(/*117.75*/i18n/*117.79*/.getMessage("train.system.chart.memoryShort")),format.raw/*117.124*/(""":</b> <span id="y">0</span>, <b>
                                                        """),_display_(/*118.58*/i18n/*118.62*/.getMessage("train.overview.charts.iteration")),format.raw/*118.108*/(""":</b> <span id="x">0</span></p>
                                                </div>
                                            </div>

                                            <!-- GPU Memory Utlization Chart -->
                                            <div class="box span6" id="gpuMemoryChart">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*125.61*/i18n/*125.65*/.getMessage("train.system.chart.gpuTitle")),format.raw/*125.107*/(""" """),format.raw/*125.108*/("""%</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <div id="gpuMemoryChartPlot" class="center" style="height: 300px;" ></div>
                                                    <p id="hoverdata"><b>"""),_display_(/*129.75*/i18n/*129.79*/.getMessage("train.system.chart.memoryShort")),format.raw/*129.124*/(""":</b> <span id="y2">0</span>, <b>
                                                        """),_display_(/*130.58*/i18n/*130.62*/.getMessage("train.overview.charts.iteration")),format.raw/*130.108*/(""":</b> <span id="x2">0</span></p>
                                                </div>
                                            </div>

                                        </div>

                                            <!-- Tables -->
                                        <div class="row-fluid">

                                                <!-- Hardware Information -->
                                            <div class="box span12">
                                                <div class="box-header">
                                                    <h2><b>"""),_display_(/*142.61*/i18n/*142.65*/.getMessage("train.system.hwTable.title")),format.raw/*142.106*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>"""),_display_(/*148.70*/i18n/*148.74*/.getMessage("train.system.hwTable.jvmCurrent")),format.raw/*148.120*/("""</th>
                                                                <th>"""),_display_(/*149.70*/i18n/*149.74*/.getMessage("train.system.hwTable.jvmMax")),format.raw/*149.116*/("""</th>
                                                                <th>"""),_display_(/*150.70*/i18n/*150.74*/.getMessage("train.system.hwTable.offHeapCurrent")),format.raw/*150.124*/("""</th>
                                                                <th>"""),_display_(/*151.70*/i18n/*151.74*/.getMessage("train.system.hwTable.offHeapMax")),format.raw/*151.120*/("""</th>
                                                                <th>"""),_display_(/*152.70*/i18n/*152.74*/.getMessage("train.system.hwTable.jvmProcs")),format.raw/*152.118*/("""</th>
                                                                <th>"""),_display_(/*153.70*/i18n/*153.74*/.getMessage("train.system.hwTable.computeDevices")),format.raw/*153.124*/("""</th>
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
                                                    <h2><b>"""),_display_(/*177.61*/i18n/*177.65*/.getMessage("train.system.swTable.title")),format.raw/*177.106*/("""</b></h2>
                                                </div>
                                                <div class="box-content">
                                                    <table class="table table-striped">
                                                        <thead>
                                                            <tr>
                                                                <th>"""),_display_(/*183.70*/i18n/*183.74*/.getMessage("train.system.swTable.hostname")),format.raw/*183.118*/("""</th>
                                                                <th>"""),_display_(/*184.70*/i18n/*184.74*/.getMessage("train.system.swTable.os")),format.raw/*184.112*/("""</th>
                                                                <th>"""),_display_(/*185.70*/i18n/*185.74*/.getMessage("train.system.swTable.osArch")),format.raw/*185.116*/("""</th>
                                                                <th>"""),_display_(/*186.70*/i18n/*186.74*/.getMessage("train.system.swTable.jvmName")),format.raw/*186.117*/("""</th>
                                                                <th>"""),_display_(/*187.70*/i18n/*187.74*/.getMessage("train.system.swTable.jvmVersion")),format.raw/*187.120*/("""</th>
                                                                <th>"""),_display_(/*188.70*/i18n/*188.74*/.getMessage("train.system.swTable.nd4jBackend")),format.raw/*188.121*/("""</th>
                                                                <th>"""),_display_(/*189.70*/i18n/*189.74*/.getMessage("train.system.swTable.nd4jDataType")),format.raw/*189.122*/("""</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            <tr>
                                                                <td id="hostName">Loading...</td>
                                                                <td id="OS">Loading...</td>
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

                                            <!-- GPU Information -->
                                        <div class="row-fluid" id="gpuTable">
                                            <div class="box span12">
                                                <div class="box-header">
                                                    <h2><b>GPU Information</b></h2>
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
        <script src="/assets/js/train/train.js"></script>   <!-- Common (lang selection, etc) -->

        <!-- Execute once on page load -->
        <script>
                $(document).ready(function () """),format.raw/*289.47*/("""{"""),format.raw/*289.48*/("""
                    """),format.raw/*290.21*/("""renderSystemPage();
                    renderTabs();
                    selectMachine();
                """),format.raw/*293.17*/("""}"""),format.raw/*293.18*/(""");
        </script>

            <!--Execute periodically (every 2 sec) -->
        <script>
                setInterval(function () """),format.raw/*298.41*/("""{"""),format.raw/*298.42*/("""
                    """),format.raw/*299.21*/("""renderSystemPage();
                """),format.raw/*300.17*/("""}"""),format.raw/*300.18*/(""", 2000);
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
                  DATE: Sun Nov 06 18:47:08 AEDT 2016
                  SOURCE: C:/DL4J/Git/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingSystem.scala.html
                  HASH: 4e6f6be77211e1b6f09b2125744807b0536ac43e
                  MATRIX: 600->1|733->39|761->41|884->138|896->142|946->172|2731->1930|2744->1934|2795->1964|3681->2822|3695->2826|3750->2859|3900->2981|3914->2985|3966->3015|4149->3170|4163->3174|4216->3205|4364->3325|4378->3329|4434->3363|6899->5801|6912->5805|6966->5838|8072->6916|8086->6920|8154->6965|8185->6966|8561->7314|8575->7318|8643->7363|8762->7454|8776->7458|8845->7504|9321->7952|9335->7956|9400->7998|9431->7999|9804->8344|9818->8348|9886->8393|10006->8485|10020->8489|10089->8535|10722->9140|10736->9144|10800->9185|11259->9616|11273->9620|11342->9666|11446->9742|11460->9746|11525->9788|11629->9864|11643->9868|11716->9918|11820->9994|11834->9998|11903->10044|12007->10120|12021->10124|12088->10168|12192->10244|12206->10248|12279->10298|13926->11917|13940->11921|14004->11962|14463->12393|14477->12397|14544->12441|14648->12517|14662->12521|14723->12559|14827->12635|14841->12639|14906->12681|15010->12757|15024->12761|15090->12804|15194->12880|15208->12884|15277->12930|15381->13006|15395->13010|15465->13057|15569->13133|15583->13137|15654->13185|22290->19792|22320->19793|22371->19815|22510->19925|22540->19926|22708->20065|22738->20066|22789->20088|22855->20125|22885->20126
                  LINES: 20->1|25->1|26->2|31->7|31->7|31->7|65->41|65->41|65->41|84->60|84->60|84->60|85->61|85->61|85->61|86->62|86->62|86->62|87->63|87->63|87->63|120->96|120->96|120->96|137->113|137->113|137->113|137->113|141->117|141->117|141->117|142->118|142->118|142->118|149->125|149->125|149->125|149->125|153->129|153->129|153->129|154->130|154->130|154->130|166->142|166->142|166->142|172->148|172->148|172->148|173->149|173->149|173->149|174->150|174->150|174->150|175->151|175->151|175->151|176->152|176->152|176->152|177->153|177->153|177->153|201->177|201->177|201->177|207->183|207->183|207->183|208->184|208->184|208->184|209->185|209->185|209->185|210->186|210->186|210->186|211->187|211->187|211->187|212->188|212->188|212->188|213->189|213->189|213->189|313->289|313->289|314->290|317->293|317->293|322->298|322->298|323->299|324->300|324->300
                  -- GENERATED --
              */
          