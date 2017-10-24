
latex_start = u"""
\\documentclass{article}
\\usepackage{makeidx}
\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}
\\usepackage{tikz}
\\usepackage{pgfplots}
\\usepackage{subfig}
\\usepackage{array}
\\usepackage{times}
\\usepackage{caption}
\\usepackage{tabularx}

\\begin{document}

\\title{%s}

\\maketitle

\\begin{abstract}
\\end{abstract}
"""

latex_end = u"""
\\end{document}
"""

latex_plot = u"""
\\begin{tikzpicture}
    \\begin{axis}[
        title={%s},
        xlabel={%s},
        ylabel={%s},
        ymajorgrids,
        xmajorgrids,
        major grid style={dashed},
        mark size=1.25,
        legend style={
            at={(0.5,-0.32)},
            anchor=north,
            legend columns=2
            }]

            \\addplot+[mark=+,error bars/.cd, y dir=both,y explicit]
            coordinates
            {
                %s
            };
            \\addlegendentry{%s}

    \\end{axis}
\\end{tikzpicture}
"""