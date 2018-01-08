import evaluaters.compare_prediction as compare_prediction
import numpy as np


def latexify_tables(tables):
    latex_tables = []

    for table_name, table in tables:
        latex_rows = ["%s \n" % table_name,
                      "\\begin{tabular}{l|c|c} \n",
                      "Model & Precision & Recall \\\\ \\hline \n"]

        for row in table:
            latex_rows.append("%s & $%.1f \\pm %.1f$ & $%.1f \\pm %.1f$ \\\\ \n" % row)

        latex_rows.append("\\end{tabular}")

        latex_tables.append("".join(latex_rows))

    return latex_tables


def stringify_tables(tables):
    string_tables = []

    for table_name, table in tables:
        string_rows = ["%s \n" % table_name,
                       "\t  |  Precision |    Recall \n"]

        for row in table:
            string_rows.append("%s | %.0f\u00B1%.0f | %.0f\u00B1%.0f \n" % row)

        string_tables.append("".join(string_rows))

    return string_tables


def latexify_multiple_evaluators_tables(tables):
    latex_tables = []

    for table_name, evaluator_names, rows in tables:
        latex_rows = ["%s \n" % table_name]

        row0 = ["\\begin{tabular}{l"]
        row1 = ["Model "]
        row2 = []
        for evaluator_name in evaluator_names:
            row0.append("|c|c")
            row1.append('& \multicolumn{{2}}{{|c}}{{{}}}'.format(evaluator_name.replace("%", "\%")))
            row2.append("& Precis & Recall ")
        row0.append("} \n")
        row1.append("\\\\ \n")
        row2.append("\\\\ \\hline \n")
        latex_rows.append("".join(row0))
        latex_rows.append("".join(row1))
        latex_rows.append("".join(row2))

        format_string = ["{:5} "]
        format_string.extend(["& ${:>3.0f} \\pm {:<2.0f}$ & ${:>3.0f} \\pm {:<2.0f}$ "
                              for _ in range(len(evaluator_names))])
        format_string.append("\\\\ \n")
        format_string = "".join(format_string)

        print(format_string)
        for row in rows:
            print(len(row))
            latex_rows.append(format_string.format(*row))

        latex_rows.append("\\end{tabular}")
        latex_tables.append("".join(latex_rows))

    return latex_tables


def stringify_multiple_evaluators_tables(tables):
    string_tables = []

    for table_name, evaluator_names, rows in tables:
        string_rows = ["%s \n" % table_name]

        row0 = ["{:^6}".format("")]
        row1 = ["{:^6}".format("")]
        for evaluator_name in evaluator_names:
            row0.append('|{:^17}'.format(evaluator_name))
            row1.append("| Precis | Recall ")
        row0.append("\n")
        row1.append("\n")
        string_rows.append("".join(row0))
        string_rows.append("".join(row1))

        format_string = ["{:5} "]
        format_string.extend(["| {:>3.0f}\u00B1{:<2.0f} | {:>3.0f}\u00B1{:<2.0f} " for _ in range(len(evaluator_names))])
        format_string.append("\n")
        format_string = "".join(format_string)

        print(format_string)
        for row in rows:
            print(len(row))
            string_rows.append(format_string.format(*row))

        string_tables.append("".join(string_rows))

    return string_tables


def precision_recall(evaluator, runs):

    precisions = []
    recalls = []

    for run in runs:
        precision, recall = compare_prediction.compare_predictions(run, evaluator)
        precisions.append(precision * 100)  # In percent
        recalls.append(recall * 100)  # In percent

    precision_mean = np.mean(precisions)
    precision_std = np.std(precisions)

    recall_mean = np.mean(recalls)
    recall_std = np.std(recalls)

    return precision_mean, precision_std, recall_mean, recall_std


class Statistics:
    def __init__(self):
        self.evaluators = []
        self.models = []

        self.evaluators.append(("Endpoint below 5", compare_prediction.endpoints_diff_below_5_overlap_over_50_percent))
        self.evaluators.append(("Overlap 25%", compare_prediction.overlap_over_25_percent))

        self.multiple_evaluators = []

        overlap_evals = [("10%", compare_prediction.overlap_over_x(10)),
                         ("25%", compare_prediction.overlap_over_x(25)),
                         ("50%", compare_prediction.overlap_over_x(50)),
                         ("75%", compare_prediction.overlap_over_x(75))]

        endpoint_evals = [("Below 10", compare_prediction.endpoint_diff_below_x(10)),
                          ("Below 5", compare_prediction.endpoint_diff_below_x(5)),
                          ("Below 2", compare_prediction.endpoint_diff_below_x(2)),
                          ("Equal 0", compare_prediction.endpoint_diff_below_x(0))]

        self.multiple_evaluators.append(("Overlap", overlap_evals))
        self.multiple_evaluators.append(("Endpoint", endpoint_evals))

    def add_model(self, model):
        self.models.append(model)

    def print_statistics(self):

        single_evaluator_tables = []
        for evaluator_name, evaluator in self.evaluators:
            rows = []
            for model_name, runs in self.models:
                row = precision_recall(evaluator, runs)
                rows.append((model_name, *row))
            single_evaluator_tables.append((evaluator_name, rows))

        latex_tables = latexify_tables(single_evaluator_tables)
        string_tables = stringify_tables(single_evaluator_tables)

        multiple_evaluator_tables = []
        for table_name, evaluators in self.multiple_evaluators:
            rows = []
            for model_name, runs in self.models:
                row = [model_name]
                for evaluator_name, evaluator in evaluators:
                    row_part = precision_recall(evaluator, runs)
                    row.extend(row_part)
                rows.append(row)
            evaluator_names = [evaluator_name for evaluator_name, evaluator in evaluators]
            multiple_evaluator_tables.append((table_name, evaluator_names, rows))

        string_tables = stringify_multiple_evaluators_tables(multiple_evaluator_tables)
        latex_tables = latexify_multiple_evaluators_tables(multiple_evaluator_tables)

        for table in latex_tables:
            print(table)

        for table in string_tables:
            print(table)

    # Assumes predictions in same order in all added models
    def print_predictions(self):

        models = [model[0] for _, model in self.models]
        if len(models) > 0:
            for i, (name, sequence, targets, _) in enumerate(models[0]):
                print(name)
                print(sequence)
                print(targets)

                for model in models:
                    print(model[i][3])

