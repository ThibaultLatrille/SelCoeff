import argparse
from glob import glob
from collections import defaultdict
import os
from libraries import format_pop

dict_method = {"MutSel": "Site-specific Mutation-Selection codon models.",
               "Omega": "Site-specific codon models.",
               "SIFT": "SIFT score",}


def minipage(size, file):
    out = "\\begin{minipage}{" + str(size) + "\\linewidth} \n"
    out += "\\includegraphics[width=\\linewidth, page=1]{" + file + "} \n"
    out += "\\end{minipage}\n"
    return out


def main(args):
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    heatmap_dict = defaultdict(list)
    for result in sorted(glob(f"{os.path.dirname(args.results)}/*.pdf")):
        _, method_lr, theta, plot, ext = os.path.basename(result).replace("_", " ").split(".")
        method = method_lr.split("-")[0]
        heatmap_dict[method].append(result)

    o = open(args.tex_include, 'w')
    o.write("\\section{Tajima's D - Fay \\& Hu's H} \n \n")
    for method, list_heatmap in heatmap_dict.items():
        o.write("\\subsection{" + dict_method[method].capitalize() + '} \n')
        o.write("\\begin{center}\n")
        for plot in list_heatmap:
            o.write("\\includegraphics[width=0.75\\linewidth, page=1]{" + plot + "} \\\\\n")
        o.write("\\end{center}\n")

    nested_dict = defaultdict(lambda: defaultdict(dict))
    for sfs in sorted(args.sfs):
        sp, pop, method_lr, ext = os.path.basename(sfs).replace("_", " ").split(".")
        method = method_lr.split("-")[0]
        nested_dict[sp][pop][method] = sfs

    for sp, nested_dict_1 in nested_dict.items():
        o.write("\\section{" + sp + "} \n \n")
        for pop, nested_dict_2 in nested_dict_1.items():
            if " " in pop:
                o.write("\\subsection{" + f"{pop} ({format_pop(pop)})" + "} \n \n")
            else:
                o.write("\\subsection{" + format_pop(pop) + "} \n \n")

            for method, sfs in nested_dict_2.items():
                o.write("\\subsubsection*{" + dict_method[method] + '} \n')
                o.write(minipage(0.49, sfs))
                o.write(minipage(0.49, sfs.replace(".pdf", ".normalize.pdf")))
                o.write("\\\\ \n")
                suffix = f"{sp}.{pop}.{method}.histogram.pdf".replace(' ', '_')
                hist_path = f"{args.hist_prefix}{suffix}"
                o.write(minipage(0.49, hist_path))
                o.write(minipage(0.35, sfs.replace("-sfs.pdf", ".polyDFE_C.pdf")))
                o.write("\\\\ \n")
    o.close()

    os.system(f"cp -f {args.tex_source} {args.tex_target}")
    output_dir = os.path.dirname(args.tex_target)
    tex_to_pdf = f"pdflatex -synctex=1 -interaction=nonstopmode -output-directory={output_dir} {args.tex_target}"
    os.system(tex_to_pdf)
    os.system(tex_to_pdf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sfs', required=False, type=str, nargs="+", dest="sfs", help="Input sfs file (pdf)")
    parser.add_argument('--hist_prefix', required=False, type=str, dest="hist_prefix", help="Input histogram folder")
    parser.add_argument('--tex_source', required=False, type=str, dest="tex_source", help="Main document source file")
    parser.add_argument('--tex_target', required=False, type=str, dest="tex_target", help="Main document target file")
    parser.add_argument('--results', required=False, type=str, dest="results", help="Results tsv file")
    parser.add_argument('--tex_include', required=False, type=str, dest="tex_include", help="Include tex file")
    main(parser.parse_args())
