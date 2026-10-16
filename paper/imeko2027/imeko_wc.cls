%%% IMEKO World Congress 2027 Latex class file
%% imeko_wc.cls
%%
%% Vers: 2.5 (2026/07/24)
%% By: Federico Tramarin (federico.tramarin@unipd.it)
%% Date: 24/07/2026
%% Notes: Cleanup of unused legacy code
%%        Improved compatibility handling for class options and shipout hooks
%%
%% Copyright (c) 2026 Federico Tramarin
%% 
%%*************************************************************************
%% Legal Notice:
%% This code is offered as-is without any warranty either expressed or
%% implied; without even the implied warranty of MERCHANTABILITY or
%% FITNESS FOR A PARTICULAR PURPOSE! 
%% User assumes all risk.
%% In no event shall the Acta Imeko or any contributor to this code be liable for
%% any damages or losses, including, but not limited to, incidental,
%% consequential, or any other damages, resulting from the use or misuse
%% of any information contained here.
%% All comments are the opinions of their respective authors and are not
%% necessarily endorsed by the Acta Imeko or Imeko.
%
% This work may be distributed and/or modified under the
% conditions of the LaTeX Project Public License, either version 1.3
% of this license or (at your option) any later version.
% The latest version of this license is in
%   https://www.latex-project.org/lppl.txt
% and version 1.3c or later is part of all distributions of LaTeX
% version 2008 or later.
%
% This work has the LPPL maintenance status `maintained'.
% 
% The Current Maintainer of this work is Federico Tramarin.
%
% This work consists of the files imeko_wc.cls, imeko_wc.bst
% and the derived file imeko_wc_template.tex and guide.tex
%
%% Retain all contribution notices and credits.
%% ** Modified files should be clearly indicated as such, including  **
%% ** renaming them and changing author support contact information. **
%%*************************************************************************

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{imeko_wc}[2026/07/24 IMEKO World Congress 2027 paper class v2.2]

%%
\newif\if@showcorr \@showcorrtrue
\DeclareOption{showcorresponding}{\@showcorrtrue}
\DeclareOption{noshowcorresponding}{\@showcorrfalse}
\DeclareOption{a4paper}{\PassOptionsToClass{a4paper}{article}}
\DeclareOption{letterpaper}{\PassOptionsToClass{letterpaper}{article}}

\DeclareOption*{%
\ClassWarningNoLine{imeko_wc}{Unknown option '\CurrentOption' ignored}%
}

%% Defaults
\ExecuteOptions{a4paper}
\ProcessOptions\relax

\PassOptionsToClass{twocolumn}{article}%

\LoadClass[10pt,notitlepage]{article}

% Ensure xcolor gets table support even if loaded indirectly by another package.
\PassOptionsToPackage{table}{xcolor}

\RequirePackage[T1]{fontenc}
\RequirePackage{iftex}
\ifPDFTeX
\RequirePackage[utf8]{inputenc}
\fi
\RequirePackage[british]{babel}
\RequirePackage{newtx}
% using Helvetica for sans-serif
\RequirePackage[scaled=0.87]{helvet}

% Enabling microtypography features, 
% both for enhanced reading
\RequirePackage{microtype} 
%\microtypesetup{tracking=true,activate=true}

\RequirePackage{ifthen}
\RequirePackage{graphicx}
\RequirePackage{xcolor}
\RequirePackage{amsmath}
\RequirePackage{caption}
\RequirePackage{fancyhdr}  % Needed to define custom headers/footers
\RequirePackage{enumitem}
\RequirePackage{array,booktabs,tabularx}
\RequirePackage{cite}
\RequirePackage{tikz}
%only needed if we want to have a double column float *mandatory* at the bottom (because latex only put them at the top by default)
\RequirePackage{stfloats}
\RequirePackage[unicode=true,
	bookmarksopen={true},
	pdffitwindow=true, 
	colorlinks=true, 
	linkcolor=ActaBlue, 
	citecolor=black, 
	urlcolor=ActaBlue, 
	hyperfootnotes=true, 
	pdfstartview={FitH},
	pdfpagemode=UseNone]{hyperref}
	
\RequirePackage[per-mode = symbol,
				exponent-product = \cdot,
				inter-unit-product = \cdot,
				uncertainty-mode = separate,
				separate-uncertainty-units = single
				]{siunitx}
\DeclareSIUnit[per-mode=symbol,per-symbol=p]{\Bps}{\byte\per\second}
\DeclareSIUnit[per-mode=symbol,per-symbol=p]{\MBps}{\mega\byte\per\second}
\DeclareSIUnit[per-mode=symbol]{\MByte}{\mega\byte}
\let\MBytes\MBps % Backward-compatible alias (historically used for MB/s)
\DeclareSIUnit[per-mode=symbol,per-symbol=p]{\bps}{bps}
\DeclareSIUnit[per-mode=symbol,per-symbol=p]{\Mbps}{\mega\bps}
\DeclareSIUnit[per-mode=symbol,per-symbol=p]{\Gbps}{\giga\bps}
\DeclareSIUnit[per-mode=symbol]{\Mbits}{\mega\bit\per\second}
\DeclareSIUnit[per-mode=symbol]{\Gbits}{\giga\bit\per\second}

%%%%%%
%
% Set page layout options (no geometry package)
%
%%%%%%

\setlength{\paperheight}{297mm}%
\setlength{\paperwidth}{210mm}%

\voffset			44pt%
\headheight 		12pt 
\headsep 			25pt
\topmargin 			-37pt %
\textheight 		232mm%
\textwidth 			170mm% 
\columnsep          6mm% 
\def\@ActaMarginE{20mm}%
\oddsidemargin\@ActaMarginE
\addtolength{\oddsidemargin}{-1in}% compensate for LaTeX's 1in offset
\evensidemargin   \@ActaMarginE
\addtolength{\evensidemargin}{-1in}% compensate for LaTeX's 1in offset

%%%%%%%%%%%%
% Font tweaks (mostly on the baseline skip) to match Word template
% Set the baseline skip to 11.25pt to match the desired line spacing for the document
\def\normalsize{\@setfontsize{\normalsize}{10}{11.6pt}}%
\normalsize
\abovedisplayskip 1.5ex plus4pt minus2pt%
\belowdisplayskip \abovedisplayskip%
\abovedisplayshortskip 0pt plus4pt%
\belowdisplayshortskip 1.5ex plus4pt minus2pt
\def\small{\@setfontsize{\small}{9}{9.8pt}}
\def\footnotesize{\@setfontsize{\footnotesize}{8}{9pt}}
\def\scriptsize{\@setfontsize{\scriptsize}{7}{7pt}}
\def\tiny{\@setfontsize{\tiny}{5}{6pt}}
\def\sublargesize{\@setfontsize{\sublargesize}{11}{12.6pt}}
\def\large{\@setfontsize{\large}{12}{14pt}}
\def\nlarge{\@setfontsize{\nlarge}{13}{15pt}}
\def\Large{\@setfontsize{\Large}{14}{17pt}}
\def\LARGE{\@setfontsize{\LARGE}{17}{20pt}}
\def\nLARGE{\@setfontsize{\nLARGE}{19}{21pt}}
\def\huge{\@setfontsize{\huge}{20.74}{24pt}}
\def\Huge{\@setfontsize{\Huge}{24}{28pt}}

%%%%%%%%%%%%
% Headers
\renewcommand{\headrulewidth}{0pt}% % No header rule
\renewcommand{\footrulewidth}{0pt}% % No footer rule
\fancyhead[L]{}%
\fancyhead[C]{}%
\fancyhead[R]{}
% Footers
% \fancyfoot[L]{\footnotesize\sffamily\selectfont {ACTA IMEKO \textbar{} www.imeko.org}}%
% \fancyfoot[C]{}%
% \fancyfoot[R]{\footnotesize\sffamily\selectfont \@JournalInfoFooter \thepage}%
\fancyfoot[L]{}%
\fancyfoot[C]{}%
\fancyfoot[R]{}%
\pagestyle{fancy} % Enables the custom headers/footers
\fancypagestyle{plain}{}

%%%%%%%%%%%%%
%% Various definitions
%Tweak first line indent width
\parindent 5mm
\flushbottom % Makes all text pages the same height
%\raggedbottom
% adapting some internal Latex parameters
% TeX default is 50
\hyphenpenalty=20
% If we didn't adjust the interword spacing, 2200 might be better.
% The TeX default is 1000
\hbadness=1000
% Acta IMEKO does not use extra spacing after punctuation
\frenchspacing
\clubpenalty=5000
\widowpenalty=2000
\displaywidowpenalty=1000

\setlength{\fboxrule}{0.75pt} % Width of the border around the abstract

% normally zero, should be relative to font height.
% put in a little rubber to help stop some bad breaks (underfull vboxes)
\parskip 0ex plus 1pt minus 1pt
%\setlength{\mathindent}{0pt}

% adjust caption properties
\captionsetup[figure]{format=plain,indention=0in,singlelinecheck=false,font={rm,small},labelsep=period,justification=justified,name=Fig.}
\captionsetup[figure*]{format=plain,indention=0in,singlelinecheck=false,font={rm,small},labelsep=period,justification=centering,name=Fig.}
\captionsetup[table]{format=plain,indention=0in,singlelinecheck=false,font={it,small},labelsep=period,justification=centering}
\def\thetable{\Roman{table}}

\setlength\abovecaptionskip{6pt plus 3pt minus 2pt}
\setlength\belowcaptionskip{0pt}% plus 3pt minus 2pt} %
\setlength\textfloatsep{12pt plus 2pt minus 2pt}
\setlength\intextsep{4pt plus 2pt minus 2pt}

\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.1}
\renewcommand{\floatpagefraction}{0.85}
\renewcommand{\dbltopfraction}{0.85}
\renewcommand{\dblfloatpagefraction}{.85}
\setcounter{topnumber}{25}
\setcounter{bottomnumber}{25}
\setcounter{totalnumber}{25}
\setcounter{dbltopnumber}{25}

%%%%%%
% Acta IMEKO item lists: we uses 5) not 5.
% a) not (a)
% iii) not iii.
% A) not A.
\def\labelenumi{\theenumi)}     \def\theenumi{\arabic{enumi}}
\def\labelenumii{\theenumii)}  \def\theenumii{\alph{enumii}}
\def\labelenumiii{\theenumiii)} \def\theenumiii{\roman{enumiii}}
\def\labelenumiv{\theenumiv)}   \def\theenumiv{\Alph{enumiv}}

% exactly the same as in article.cls
\def\p@enumii{\theenumi}
\def\p@enumiii{\theenumi(\theenumii)}
\def\p@enumiv{\p@enumiii\theenumiii}

\setlist{nosep}
\newlist{enumerate-a}{enumerate}{1}
\setlist[enumerate,1,2]{left= \parindent, labelsep=0.5em}
\setlist[enumerate-a,1,2]{label=\alph*., left= \parindent, labelsep=0.5em}
\setlist[itemize,1]{label=-, left= \parindent, labelsep=0.5em}

%%%%%%%%%%
% SECTIONING

\def\@seccntformat#1{\csname the#1\endcsname.\space}
\def\thesection{\Roman{section}}
\def\thesubsection{\Alph{subsection}}
\def\thesubsubsection{\thesubsection.\arabic{subsubsection}}
% Settings for titles of sections and subsections

% arguments are: #1 - sectiontype name
% (for \@sect)   #2 - section level
%                #3 - section heading indent
%                #4 - top separation (absolute value used, neg indicates not to indent main text)
%                     If negative, make stretch parts negative too!
%                #5 - (absolute value used) positive: bottom separation after heading,
%                      negative: amount to indent main text after heading
%                Both #4 and #5 negative means to indent main text and use negative top separation
%                #6 - font control
\def\section{\@startsection{section}{1}{\z@}{12pt plus 0.1ex minus 0.5ex}{3pt plus 0ex minus 0.1ex}{\centering\normalsize\MakeUppercase}}%

\def\subsection{\@startsection{subsection}{2}{\parindent}{9pt plus 1pt minus 1pt}{3pt plus 1pt minus 1pt}{\normalsize\itshape}}%

\def\subsubsection{\@startsection{subsubsection}{3}{\z@}{9pt plus 1pt minus 1pt}{1pt plus 1pt minus 1pt}{\normalsize}}%

\def\paragraph{\@startsection{paragraph}{4}{\parindent}{0ex plus 0.1ex minus 0.1ex}{-1ex}{\small\itshape}}%

%%%%%%%%%%%%%%
%% tuning tables
% set tabular/tables to use serif small fonts
\renewcommand{\tabular}{\let\@halignto\@empty%\sffamily%
	%\arraystretch=1.15%
	\small%
	%\rowcolors{1}{gray!19}{white}%
	\@tabular}

\AtBeginEnvironment{tabularx}{%
	\small%
	\rowcolors{2}{white}{gray!19}%
}
\newcolumntype{C}{>{\centering\arraybackslash}X}

%%%%%%%%%%%%%
\definecolor{ActaBlue}{RGB}{14,99,189}
\definecolor{AIAbstractBackground}{RGB}{199,217,240} % Color of the boxes behind the abstract and headings
\definecolor{TableGray}{RGB}{230,231,231} % Alternating row background colour

% Definitions for ABSTRACT+AUTHOR FRAME
%----------------------------------------------------------------------
\def\@Keywords{No Keywords.}
\def\@Location{Rimini, Italy}
\def\@Dates{August 30 - September 3, 2027}

\newcommand{\Archive}[1]{\edef\@Archive{#1}}
\newcommand{\Abstract}[1]{\edef\@Abstract{#1}}
\newcommand{\Keywords}[1]{\def\@Keywords{#1}}
\newcommand{\Funding}[1]{\ifthenelse{\equal{#1}{}}{\def\@Funding{No funding specified.}}{\edef\@Funding{#1}}}
%%%% the following snippet of code is needed to set the proper PDF Metadata at 
%%%% runtime, by taking the appropriate values from the source TeX file
% The following line is needed to set the title. The internal \@title is already defined internally 
% by the latex kernel, but unfortunately the \maketitle command will clean its value on exit.
% With this redefinition we can set the title in the PDF metadata.
\def\title#1{\gdef\@title{#1}\gdef\PaperTitle{#1}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors Management
% Taken inspiration from Elsevier elsarticle and largely simplified
% for the Acta Imeko class
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Definitions to manage multiple authors
% Definitions for management of multiple authors and affiliations
\def\author{\@@authori}
%\def\author{\@ifnextchar[{\@@author}{\@author}}
\def\@AIaffiliations{}
\def\@AIauthors{}
\def\@AIauthorsNames{}
\def\@AIauthorsNamesCitation{}

%some temporary variables for internal use
\let\authorsep\@empty
\let\sep\@empty
\let\@cormark\@empty
\def\CorrespondingAuthor{}
\def\@CorrespondingAuthorName{}

\def\@author@name{}
\def\@author@surname{}
\def\@authorshortname{}

\def\@tmpauthname{}

\define@key{authornames}{name}{%
	\def\@author@name{#1}}
%	\g@addto@macro\@tmpauthname{#1}}
\define@key{authornames}{surname}{%
	\def\@author@surname{#1}}
%	\g@addto@macro\@tmpauthname{\space#1}}
\define@key{authornames}{abbreviation}{%
	\def\@authorshortname{#1}}

\newif\ifnotyetprocessed
\newif\if@AIauthoriskeyval
\def\@@authori[#1]#2{%
	\def\@author@name{}%
	\def\@author@surname{}%
	\def\@authorshortname{}%
	\@AIauthoriskeyvalfalse
	\in@{=}{#2}%
	\ifin@
	  \@AIauthoriskeyvaltrue
    \setkeys{authornames}{#2}%
	  \ifx\@author@name\@empty
		  \ClassError{imeko_wc}{Missing required author key 'name'}{Use \string\author[aff]{name=...,surname=...,abbreviation=...} or \string\author[aff]{Full Name}.}%
	  \fi
	  \ifx\@author@surname\@empty
		  \ClassError{imeko_wc}{Missing required author key 'surname'}{Use \string\author[aff]{name=...,surname=...,abbreviation=...} or \string\author[aff]{Full Name}.}%
	  \fi
	\else
	  \def\@author@name{#2}%
	  \def\@authorshortname{#2}%
	\fi
	\ifx\@authorshortname\@empty
		\ClassWarning{imeko_wc}{Missing author key 'abbreviation'; using full name for citation}%
		\edef\@authorshortname{\@author@name\space\@author@surname}%
	\fi
  \begingroup
  \toks@{\endgroup\processauthor{#1}}%
  \protected@edef\tmpauthname{%
    \the\toks@{\@author@name\space\@author@surname}{\@authorshortname}%
  }%
  \tmpauthname
}%

\newcounter{auth}
\long\def\processauthor#1#2#3{%
  \g@addto@macro\@AIauthors{%
    \def\baselinestretch{1}%
    \refstepcounter{auth}% increment by 1
    \notyetprocessedtrue
    \authorsep#2\unskip
    \textsuperscript{%
      \@for\@@affmark:=#1\do{%
        \edef\affnum{%
          \@ifundefined{X@\@@affmark}{affn?}{\AIrefMark{\@@affmark}}%
        }%
        \unskip\sep\affnum
        \ifnotyetprocessed
          \NewAIaffLabel{auth-\theauth}{+}%
        \fi
        \let\sep=,%
      }%
      \edef\cormark{%
        \@ifundefined{X@cor-\theauth}{}{%
          \if@showcorr
            {,\AIrefMark{cor-\theauth}}%
          \else
            {}%
          \fi
        }%
      }%
      \unskip\cormark
      \ifx\cormark\@empty
      \else
        \ifnotyetprocessed % if the author is corresponding and is the first time we are here
          \g@addto@macro\@CorrespondingAuthorName{#2}% add the author name to the output
          \notyetprocessedfalse % if it is a subsequent round, do nothing
        \fi
      \fi
    }%
    \gdef\authorsep{\unskip,\space}%
    \global\let\sep\@empty
  }%%
  \ifx\@AIauthorsNames\@empty
    \g@addto@macro\@AIauthorsNames{#2}%
  \else
    \g@addto@macro\@AIauthorsNames{;\space#2}%
  \fi
  \ifx\@AIauthorsNamesCitation\@empty
	\g@addto@macro\@AIauthorsNamesCitation{#3}%
  \else
	\g@addto@macro\@AIauthorsNamesCitation{,\space#3}%
  \fi
}%

% we  process all the affiliations, creating an affLabel for each one
% and then writing them down
% in this phase, at the first round, we hence create the variables for cross referencing with 
% authors, in the second round
\def\affiliation{\@@affiliation}
\newcounter{affn}
\renewcommand\theaffn{\arabic{affn}}
\long\def\@@affiliation[#1]#2{%
	\g@addto@macro\@AIaffiliations{%
		\def\baselinestretch{1}%
		\refstepcounter{affn}%
		%\xdef\@currentlabel{\theaffn}%
		%\typeout{Federico - @@affiliation macro -> \@currentlabel}
		\AIaffLabel{#1}%here we create a first round of refmarks for each affiliation
		{\upshape\textsuperscript{\theaffn}}\space#2\par\vspace{1pt}}% the vspace is necessary to keep the same interline spacing of word (due to superscripts)
	}

% AIaffLabel is used in the first round to create a global variable for affiliation referencing
% that is written to the aux file
% in practice, each affiliation mark (it can be a number, a letter, etc) is
% associated with a sequence number of the affiliation
\def\AIaffLabel#1{\@bsphack\protected@write\@auxout{}%
	{\string\NewAIaffLabel{#1}{\@currentlabel}}\@esphack}
\def\NewAIaffLabel#1#2{\expandafter\xdef\csname X@#1\endcsname{#2}}

% AIrefMark is used in the second round to xref the previous defined variables
\def\AIrefMark#1{\@ifundefined{X@#1}{0}{\csname X@#1\endcsname}%
}

% here we create a first round of refmarks for the corresponding author
\def\CorrespondingAuthorNumber#1{\NewAIaffLabel{cor-#1}{$\ast$}}

\newcounter{cnote}
\def\CorrespondingAuthorEmail#1{\refstepcounter{cnote}%
	\AIaffLabel{#1}%
	\g@addto@macro\CorrespondingAuthor{%
	\@CorrespondingAuthorName, e-mail: \url{#1}}}

%%%% URL
\RequirePackage{xurl}
\urlstyle{same}
% Line breaks in URL
\def\UrlDigits{\do\1\do\2\do\3\do\4\do\5\do\6\do\7\do\8\do\9\do\0}
\g@addto@macro{\UrlBreaks}{\UrlOrds}
\g@addto@macro{\UrlBreaks}{\UrlDigits}

%% CITATION COMMANDS
%% 
%% F.Tramarin: The following lines are directly taken from IEEEtran.cls
%% 				it was useless to reinvent the wheel...they are simply brilliant!
%%
%% From IEEEtran.cls
%% V1.6 no longer supports the older, nonstandard \shortcite and \citename setup stuff
% Modify Latex2e \@citex to separate citations with "], ["
\def\@citex[#1]#2{%
	\let\@citea\@empty
	\@cite{\@for\@citeb:=#2\do
		{\@citea\def\@citea{], [}%
			\edef\@citeb{\expandafter\@firstofone\@citeb\@empty}%
			\if@filesw\immediate\write\@auxout{\string\citation{\@citeb}}\fi
			\@ifundefined{b@\@citeb}{\mbox{\reset@font\bfseries ?}%
				\G@refundefinedtrue
				\@latex@warning
				{Citation `\@citeb' on page \thepage \space undefined}}%
			{\hbox{\csname b@\@citeb\endcsname}}}}{#1}}

% Acta IMEKO needs citation "sorting" and also grouping of three or
% more citations.
% To this extent, we use Donald Arseneau's cite.sty package. 
% As defined in IEEEtran (Thanks again) cite.sty will notice that the
% following format controls are already defined and will not
% redefine them.
\def\citepunct{], [}
\def\citedash{]--[}

% ---------------------------------------------------------------------
\def\@Journal{IMEKO World Congress 2027}


\def\abstract{\normalfont%
      \normalsize\bfseries\textit{Abstract}---\relax}
\def\endabstract{\relax\vspace{10pt}\par
    \normalfont\normalsize}
\long\def\AIWC@setandprintkeywords#1{%
	\gdef\@Keywords{#1}%
	\@ifpackageloaded{hyperref}{\hypersetup{pdfkeywords={\@Keywords}}}{}%
	\normalfont\normalsize\bfseries\textit{Keywords}---\relax
	\ignorespaces#1\unskip\relax\par\vspace{1pt}
	\normalfont\normalsize}
\@ifundefined{NewDocumentEnvironment}{%
	\newenvironment{keywords}{\collect@body\AIWC@setandprintkeywords}{}
}{%
	\NewDocumentEnvironment{keywords}{+b}{\AIWC@setandprintkeywords{#1}}{}
}

\newcommand{\AIWCFirstPageBanner}{%
	\setlength{\unitlength}{1mm}%
	\begin{picture}(\textwidth,40)
		\IfFileExists{logo_IMEKO2027_v1.png}{%
			\put(173,-15){\includegraphics[width=1.68cm]{logo_IMEKO2027_v1.png}}%
		}{%
			\ClassWarningNoLine{imeko_wc}{File 'logo_IMEKO2027_v1.png' not found; skipping logo on first page}%
		}%
		\IfFileExists{imeko.png}{%
			\put(20,-15){\includegraphics[height=1.5cm]{imeko.png}}%
		}{%
			\ClassWarningNoLine{imeko_wc}{File 'imeko.png' not found; skipping logo on first page}%
		}%
		\put(35,-16){\makebox(150,15)[tl]{
			\parbox{100mm}{\footnotesize XXV IMEKO World Congress\par
	 	“Metrology for Humanity”\par
		August 30 - September 3, 2027\par
		Rimini, Italy}}}
	\end{picture}
}

% Compatibility fallback for older LaTeX kernels without the hook interface.
\@ifundefined{AddToHookNext}{%
	\RequirePackage{eso-pic}%
	\AddToShipoutPictureBG*{\AIWCFirstPageBanner}%
}{%
	\AddToHookNext{shipout/background}{\AIWCFirstPageBanner}%
}

\def\@funding{}

% user command to invoke the title page
\def\maketitle{\par%
  \begingroup%
  \normalfont%
  \def\thefootnote{}%
  \def\footnotemark{}% and kill space from \thanks within author
  \let\@makefnmark\relax
  \long\def\@makefntext##1{\setlength{\parindent}{4mm}##1}
  \footnotesize%       equal spacing between thanks lines
  \footnotesep 0.7\baselineskip%see global setting of \footnotesep for more info
  \normalsize%  
           \twocolumn[\@maketitleWC]%
		 	\thispagestyle{empty}\@thanks\begin{NoHyper}\footnotetext{\@funding}\end{NoHyper}%
		 %
  \endgroup
	\hypersetup{
	pdftitle={\PaperTitle},
	pdfsubject={\@Journal,\space\@Location,\space\@Dates\space(1-\@abspage@last)},%
	pdfkeywords={\@Keywords},
	pdfauthor={\@AIauthorsNamesCitation},
	pdfinfo={
	Company={IMEKO - The International Measurement Confederation},
	IMEKO World Congress 27 Authors={\@AIauthorsNames},
	IMEKO World Congress 27 Location={\@Location},
	IMEKO World Congress 27 Dates={\@Dates},
	IMEKO World Congress 27 Page Numbers={1-\@abspage@last},%
	IMEKO World Congress 27 Copyright={This is an open-access article distributed under %
the terms of the Creative Commons Attribution 4.0 International License - https://creativecommons.org/licenses/by/4.0/}
	}
	}%
  \setcounter{footnote}{0}\let\maketitle\relax\let\@maketitleWC\relax
  \gdef\@thanks{}%
  % v1.6b do not clear these as we will need the title again for peer review papers
  % \gdef\@author{}\gdef\@title{}%
  \let\thanks\relax}

\long\def\funding#1{\def\@funding{#1}}

\def\@maketitleWC{\newpage\null\vskip-7pt%%
\bgroup%\par
\centering%
   \vskip0mm{\Huge\@title\par}\relax%
   \vskip13pt\par%
   % V1.6 handle \author differently if in conference mode
	%\lineskip.5em
	\sublargesize{\@AIauthors\par\addvspace{5pt}%
				\emph{\@AIaffiliations}\par%
				\emph{*Corresponding Author:~\CorrespondingAuthor}}%\relax
\par\addvspace{11pt}\egroup%}
}

%----------------------------------------------------------------------
%	REFERENCES
%----------------------------------------------------------------------  

% In a similar way to IEEEtran, we provide a command to break the last column
% to try to equalize le bibliography on the last page
\edef\@AItriggerrefnum{0}   % the default of zero means that the command is not executed
\def\@AItriggercmd{\newpage}
\newcount\@AItmpcount
\def\AIBreakAtRef#1{\@AItmpcount=#1%
\edef\@AItriggerrefnum{\the\@AItmpcount}}%

% trigger command at the given reference
\def\@AIbibitemprefix{\@AItmpcount=\@AItriggerrefnum\relax%
\advance\@AItmpcount by -1\relax%
\ifnum\c@enumiv=\@AItmpcount\relax\@AItriggercmd\relax\fi}

% controls bib item spacing
%\def\AIbibitemsep{0pt plus .5pt}
\def\AIbibitemsep{0pt}
\setlength{\labelsep}{2mm}
% Remove brackets from numbering in List of References
\renewcommand{\@biblabel}[1]{[#1]\hfill}

\def\thebibliography#1{\section*{\refname}%
    \addcontentsline{toc}{section}{\refname}%
    \footnotesize\list{\@biblabel{\@arabic\c@enumiv}}%
    {\settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth%
    \advance\leftmargin\labelsep\relax
    \itemsep \AIbibitemsep\relax
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \renewcommand\theenumiv{\@arabic\c@enumiv}}%
    \let\@AIlatexbibitem\bibitem%
    \def\bibitem{\@AIbibitemprefix\@AIlatexbibitem}%
\def\newblock{\hskip .11em plus .33em minus .07em}%
\setlength\parskip{1pt plus 1pt minus .5pt}\relax%
\sloppy\clubpenalty4000\widowpenalty4000\interlinepenalty500%
\sfcode`\.=1000}

\def\endthebibliography{%
  \def\@noitemerr{\@latex@warning{Empty `thebibliography' environment}}%
  \endlist
}

\endinput
%%
%% End of file `imeko_wc.cls'.
