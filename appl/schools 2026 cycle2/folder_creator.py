#!/usr/bin/env python3
"""Create one folder per school in the current directory, each containing
blank statement-of-purpose and personal-statement .tex files with headings."""

from pathlib import Path

AUTHOR = "Ian Chi"

# folder name -> (full university name, department)
SCHOOLS = {
    "Boston_U": ("Boston University", "Department of Computer Science"),
    "MIT": ("Massachusetts Institute of Technology",
            "Department of Electrical Engineering and Computer Science"),
    "UC_Berkeley": ("University of California, Berkeley",
                    "Department of Electrical Engineering and Computer Sciences"),
    "Waterloo": ("University of Waterloo", "David R. Cheriton School of Computer Science"),
    "Stanford": ("Stanford University", "Department of Computer Science"),
    "CMU": ("Carnegie Mellon University", "Computer Science Department"),
    "NYU": ("New York University", "Courant Institute, Department of Computer Science"),
    "Columbia": ("Columbia University", "Department of Computer Science"),
    "Cornell": ("Cornell University", "Department of Computer Science"),
    "Dartmouth": ("Dartmouth College", "Department of Computer Science"),
    "U_Alberta": ("University of Alberta", "Department of Computing Science"),
    "UCD": ("University of California, Davis", "Department of Computer Science"),
    "UCI": ("University of California, Irvine", "Department of Computer Science"),
    "UCSD": ("University of California, San Diego",
             "Department of Computer Science and Engineering"),
    "UIUC": ("University of Illinois Urbana-Champaign",
             "Siebel School of Computing and Data Science"),
    "UPenn": ("University of Pennsylvania",
              "Department of Computer and Information Science"),
    "USC": ("University of Southern California",
            "Thomas Lord Department of Computer Science"),
    "UCR": ("University of California, Riverside",
            "Department of Computer Science and Engineering"),
    "UCSC": ("University of California, Santa Cruz",
             "Department of Computer Science and Engineering"),
    "Simon_Fraser": ("Simon Fraser University", "School of Computing Science"),
}

DOCUMENTS = {
    "sop.tex": "Statement of Purpose",
    "personal_statement.tex": "Personal Statement",
}

TEMPLATE = r"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}

\begin{{document}}

\begin{{center}}
    {{\Large \textbf{{{title}}}}}\\[4pt]
    {{\large {author}}}\\[2pt]
    Ph.D. Applicant, {department}\\
    {university}
\end{{center}}

\vspace{{1em}}

% Write here.

\end{{document}}
"""


def main():
    root = Path(".")
    for folder, (university, department) in SCHOOLS.items():
        school_dir = root / folder
        school_dir.mkdir(exist_ok=True)
        for filename, title in DOCUMENTS.items():
            path = school_dir / filename
            if path.exists():
                print(f"skip (exists): {path}")
                continue
            path.write_text(TEMPLATE.format(
                title=title, author=AUTHOR,
                department=department, university=university))
            print(f"created: {path}")


if __name__ == "__main__":
    main()