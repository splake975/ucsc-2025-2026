\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{curriculum-vitae}[2022/05/24 Curriculum Vitae (CV) class]

\LoadClass[12pt, letterpaper]{article} 


%%%%%%%%%%%%%%%%%%%%%
%requires hyperref
%%%%%%%%%%%%%%%%%%%%%

% ANCHOR Page layout
\RequirePackage[empty]{fullpage}
\addtolength{\oddsidemargin}{-0.530in}
\addtolength{\evensidemargin}{-0.375in}
\addtolength{\textwidth}{1in}
\addtolength{\topmargin}{-.45in}
\addtolength{\textheight}{1in}

\setlength\parindent{0pt}

% ANCHOR Section formatting
\RequirePackage{titlesec}
\titleformat{\section}{
  \vspace{-10pt}\scshape\raggedright\large
}{}{0em}{}[\titlerule \vspace{-6pt}]

% ANCHOR Enumerate
\RequirePackage{enumitem}
\setlist[itemize]{leftmargin=*}

\RequirePackage{array}

\newcolumntype{L}[1]{>{\raggedright\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\newcolumntype{C}[1]{>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}
\newcolumntype{R}[1]{>{\raggedleft\let\newline\\\arraybackslash\hspace{0pt}}m{#1}}

%% \itemheading{title}{subtitle}{date}
\newcommand{\itemheading}[3]{
    \normalsize
    \vspace{-1pt}
    \begin{tabular*}{0.97\textwidth}[t]{L{0.71\textwidth} R{0.25\textwidth}}
      \textbf{#1} & \textit{#3} \\
      \itshape #2 & \\
    \end{tabular*}
    \vspace{-4pt}
    \par
}

\newenvironment{itemfootnote}{
    \vspace{10pt}
    \setlength{\parskip}{3pt}
    \footnotesize
    \sl
}{
    \par
    \vspace{-4pt}
}


%% \begin{itemshortlist}{title}
%% \end{itemshortlist}
\newenvironment{itemshortlist}[1]{
    \normalsize
    \makebox[7em]{\textbf{#1}:\hfill}
}{
    \vspace{-3pt}
}

% ANCHOR Options to package
\PassOptionsToPackage{hidelinks}{hyperref}

% ANCHOR Make title

%% \email{}
%% \phone{}
\def\email#1{\gdef\@email{#1}}
\def\phone#1{\gdef\@phone{#1}}
\def\site#1{\gdef\@site{#1}}


\AtBeginDocument{
    \renewcommand{\maketitle}{
        \setlength{\tabcolsep}{0in}
        \begin{tabular*}{\textwidth}{l@{\extracolsep{\fill}}r}
            \textbf{{\LARGE \@author}} & Email: %\hyperref[mailto:\@emaill]
            {\@email} \\
            & Phone: \@phone\\
            & Website: %\hyperref[\@site]
            {\@site}
        \end{tabular*}
    }
}
