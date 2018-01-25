import model.hyper_params_search as hyper_params_search
import model.cross_validation as cross_validation
import experiments


if __name__ == '__main__':
    # experiments.test()
    # experiments.compare_datasets()
    # hyper_params_search.do_hyper_params_search()
    # cross_validation.do_3_fold_cross_validation()

    statistics_dataset_size = experiments.dataset_size_test()
    # statistics_units = experiments.test_hyperparams("units")
    # statistics_l2_beta = experiments.test_hyperparams("l2_beta")

    statistics_dataset_size.print_statistics()
    # statistics_units.print_statistics()
    # statistics_l2_beta.print_predictions()
