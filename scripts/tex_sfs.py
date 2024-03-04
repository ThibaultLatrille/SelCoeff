import argparse
from glob import glob
from collections import defaultdict
import os
from libraries import format_pop, sample_list_dico

dict_method = {"MutSel": "Site-specific Mutation-Selection codon models.",
               "Omega": "Site-specific codon models.",
               "SIFT": "SIFT score", }


def minipage(size, file, name=""):
    assert os.path.exists(file), f"File {file} does not exist"
    out = "\\begin{minipage}{" + str(size) + "\\linewidth} \n"
    out += "\\flushleft {\\tiny " + name + "}\n"
    out += "\\includegraphics[width=\\linewidth, page=1]{" + file + "} \n"
    out += "\\end{minipage}\n"
    return out


def main(args):
    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    sample_dico = sample_list_dico(args.sample_list)
    heatmap_dict = defaultdict(list)
    for result in sorted(glob(f"{os.path.dirname(args.results)}/*.pdf")):
        _, method_lr, theta, plot, ext = os.path.basename(result).replace("_", " ").split(".")
        method = method_lr.split("-")[0]
        heatmap_dict[method].append(result)

    nested_dict = defaultdict(lambda: defaultdict(dict))
    for sfs in sorted(args.sfs):
        sp, pop, method_lr, ext = os.path.basename(sfs).replace("_", " ").split(".")
        method = method_lr.split("-")[0]
        nested_dict[method][sp][pop] = sfs

    o = open(args.tex_include, 'w')
    for method, nested_dict_1 in nested_dict.items():
        o.write("\\section{" + dict_method[method].capitalize() + '} \n')

        list_heatmap = heatmap_dict[method]
        o.write("\\subsection{All populations} \n")
        o.write("\\begin{center}\n")
        for plot in list_heatmap:
            o.write("\\includegraphics[width=0.95\\linewidth, page=1]{" + plot + "} \\\\\n")
        o.write("\\end{center}\n")

        for sp, nested_dict_2 in nested_dict_1.items():
            o.write("\\section{" + (sp.split(" ")[0] if sp in ["Capra hircus", "Ovis aries"] else sp) + "} \n \n")
            for pop, sfs in nested_dict_2.items():
                if len(nested_dict_2) > 1:
                    o.write("\\subsection{" + sample_dico[format_pop(pop)] + "} \n \n")

                dfe_suffix = f"{sp}.{pop}.{method}.pdf".replace(' ', '_')
                dfe_path = f"{args.dfe_prefix}{dfe_suffix}"

                six_panels = "/HOLD_" in args.hist_prefix
                if six_panels:
                    hist_holder = f"{sp}.{pop}.".replace(' ', '_')
                    hist_subs_path = args.hist_prefix.replace("/HOLD_", "/subs." + hist_holder) + ".pdf"
                    hist_poly_path = args.hist_prefix.replace("/HOLD_", "/poly." + hist_holder) + ".pdf"

                    o.write(minipage(0.32, dfe_path, "A: Expected DFE for mutations"))
                    o.write(minipage(0.32, dfe_path.replace(".pdf", ".Flow.pdf"),
                                     "B: Expected DFE for substitutions"))
                    o.write(minipage(0.32, hist_subs_path, "C: Observed DFE for substitutions"))

                    o.write("\\\\ \n")
                    o.write(minipage(0.32, hist_poly_path, "D: Observed DFE for polymorphisms"))
                    o.write(minipage(0.32, sfs.replace("-sfs.pdf", "-sfs.normalize.pdf"),
                                     "E: Site frequency spectrum"))
                    for model in ["C", "D"]:
                        polyDFE = sfs.replace("-sfs.pdf", f".polyDFE_{model}.pdf")
                        if os.path.exists(polyDFE):
                            o.write(minipage(0.32, polyDFE,
                                             f"F: $S$ as function of $S_0$ for each class"))
                else:
                    hist_suffix = f"{sp}.{pop}.pdf".replace(' ', '_')
                    hist_path = f"{args.hist_prefix}{hist_suffix}"
                    o.write(minipage(0.49, dfe_path))
                    o.write(minipage(0.49, hist_path))
                    o.write("\\\\ \n")
                    o.write(minipage(0.49, sfs.replace("-sfs.pdf", "-sfs.pdf")))
                    for model in ["C", "D"]:
                        polyDFE = sfs.replace("-sfs.pdf", f".polyDFE_{model}.pdf")
                        if os.path.exists(polyDFE):
                            o.write(minipage(0.49, polyDFE))
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
    parser.add_argument('--dfe_prefix', required=False, type=str, dest="dfe_prefix", help="Input dfe folder")
    parser.add_argument('--tex_source', required=False, type=str, dest="tex_source", help="Main document source file")
    parser.add_argument('--tex_target', required=False, type=str, dest="tex_target", help="Main document target file")
    parser.add_argument('--results', required=False, type=str, dest="results", help="Results tsv file")
    parser.add_argument('--tex_include', required=False, type=str, dest="tex_include", help="Include tex file")
    parser.add_argument('--sample_list', required=False, type=str, dest="sample_list", help="Sample list file")
    main(parser.parse_args())
