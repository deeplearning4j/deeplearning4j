
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

<!--~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ~ Copyright (c) 2015-2018 Skymind, Inc.
  ~
  ~ This program and the accompanying materials are made available under the
  ~ terms of the Apache License, Version 2.0 which is available at
  ~ https://www.apache.org/licenses/LICENSE-2.0.
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
  ~ WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
  ~ License for the specific language governing permissions and limitations
  ~ under the License.
  ~
  ~ SPDX-License-Identifier: Apache-2.0
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~-->

<html lang="en">
    <head>

        <meta charset="utf-8">
        <title>"""),_display_(/*24.17*/i18n/*24.21*/.getMessage("train.pagetitle")),format.raw/*24.51*/("""</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <link rel="stylesheet" href="/assets/webjars/coreui__coreui/2.1.9/dist/css/coreui.min.css">


        <script src="http://code.jquery.com/jquery-3.3.1.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js"></script>
        <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/js/bootstrap.min.js"></script>
        <script src="/assets/webjars/coreui__coreui/2.1.9/dist/js/coreui.min.js"></script>

        <link rel="shortcut icon" href="/assets/img/favicon.ico">
    </head>

    <body class="app sidebar-show aside-menu-show">
        <header class="app-header navbar">
                <!-- Header content here -->
            HEADER
        </header>
        <div class="app-body">
            <div class="sidebar">
                <nav class="sidebar-nav">
                    <ul class="nav">
                        <li class="nav-item"><a class="nav-link" href="javascript:void(0);"><i class="icon-bar-chart"></i>"""),_display_(/*47.124*/i18n/*47.128*/.getMessage("train.nav.overview")),format.raw/*47.161*/("""</a></li>
                        <li class="nav-item"><a class="nav-link" href="model"><i class="icon-tasks"></i>"""),_display_(/*48.106*/i18n/*48.110*/.getMessage("train.nav.model")),format.raw/*48.140*/("""</a></li>
                        <li class="nav-item"><a class="nav-link" href="system"><i class="icon-dashboard"></i>"""),_display_(/*49.111*/i18n/*49.115*/.getMessage("train.nav.system")),format.raw/*49.146*/("""</a></li>
                        <li class="nav-item nav-dropdown">
                            <a class="nav-link nav-dropdown-toggle" href="#">
                                <i class="nav-icon cui-puzzle"></i> """),_display_(/*52.70*/i18n/*52.74*/.getMessage("train.nav.language")),format.raw/*52.107*/("""
                            """),format.raw/*53.29*/("""</a>
                            <ul class="nav-dropdown-items">
                                <li class="nav-item"><a class="submenu" href="javascript:void(0);" onclick="languageSelect('en', 'overview')"><i class="icon-file-alt"></i>English</a></li>
                                <li class="nav-item"><a class="submenu" href="javascript:void(0);" onclick="languageSelect('de', 'overview')"><i class="icon-file-alt"></i>Deutsch</a></li>
                                <li class="nav-item"><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ja', 'overview')"><i class="icon-file-alt"></i>日本語</a></li>
                                <li class="nav-item"><a class="submenu" href="javascript:void(0);" onclick="languageSelect('zh', 'overview')"><i class="icon-file-alt"></i>中文</a></li>
                                <li class="nav-item"><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ko', 'overview')"><i class="icon-file-alt"></i>한글</a></li>
                                <li class="nav-item"><a class="submenu" href="javascript:void(0);" onclick="languageSelect('ru', 'overview')"><i class="icon-file-alt"></i>русский</a></li>
                            </ul>
                        </li>
                    </ul>

"""),format.raw/*65.188*/("""
"""),format.raw/*66.167*/("""
"""),format.raw/*67.173*/("""
"""),format.raw/*68.156*/("""
"""),format.raw/*69.59*/("""
"""),format.raw/*70.190*/("""
"""),format.raw/*71.45*/("""
"""),format.raw/*72.183*/("""
"""),format.raw/*73.60*/("""
"""),format.raw/*74.184*/("""
"""),format.raw/*75.60*/("""
"""),format.raw/*76.183*/("""
"""),format.raw/*77.56*/("""
"""),format.raw/*78.183*/("""
"""),format.raw/*79.55*/("""
"""),format.raw/*80.183*/("""
"""),format.raw/*81.55*/("""
"""),format.raw/*82.183*/("""
"""),format.raw/*83.60*/("""
"""),format.raw/*84.34*/("""
"""),format.raw/*85.30*/("""
                """),format.raw/*86.17*/("""</nav>
            </div>
            <main class="main">
                    <!-- Main content here -->
                MAIN
            </main>
        </div>
        <footer class="app-footer">
                <!-- Footer content here -->
            FOOTER
        </footer>
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
                  DATE: Tue May 07 15:03:05 AEST 2019
                  SOURCE: c:/DL4J/Git/deeplearning4j/deeplearning4j/deeplearning4j-ui-parent/deeplearning4j-play/src/main/views/org/deeplearning4j/ui/views/training/TrainingOverview.scala.html
                  HASH: 7d0cc97442516f114fe7639bd31eab1bf8ccbe84
                  MATRIX: 604->1|737->39|765->41|1683->932|1696->936|1747->966|2886->2077|2900->2081|2955->2114|3099->2230|3113->2234|3165->2264|3314->2385|3328->2389|3381->2420|3627->2639|3640->2643|3695->2676|3753->2706|5064->4175|5095->4343|5126->4517|5157->4674|5187->4734|5218->4925|5248->4971|5279->5155|5309->5216|5340->5401|5370->5462|5401->5646|5431->5703|5462->5887|5492->5943|5523->6127|5553->6183|5584->6367|5614->6428|5644->6463|5674->6494|5720->6512
                  LINES: 20->1|25->1|26->2|48->24|48->24|48->24|71->47|71->47|71->47|72->48|72->48|72->48|73->49|73->49|73->49|76->52|76->52|76->52|77->53|89->65|90->66|91->67|92->68|93->69|94->70|95->71|96->72|97->73|98->74|99->75|100->76|101->77|102->78|103->79|104->80|105->81|106->82|107->83|108->84|109->85|110->86
                  -- GENERATED --
              */
          