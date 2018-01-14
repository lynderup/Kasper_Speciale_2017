import time

import model.joint_model as joint_model
import dataprovider.joint_dataprovider as joint_dataprovider
import model.util as util
import decoder.decoder as decoder
import model.hyper_params_search as hyper_params_search
import model.cross_validation as cross_validation
import hmm.hmm_main as hmm

from evaluaters.statistics import Statistics

step1_config = joint_model.StepConfig(batch_size=10,
                                      num_input_classes=20,
                                      num_output_classes=2,
                                      starting_learning_rate=0.01,
                                      decay_steps=10,
                                      decay_rate=0.99,
                                      num_units=50,  # 50
                                      train_steps=1000,  # 1000
                                      keep_prop=0.5,
                                      l2_beta=0.001)

# step3_config = joint_model.StepConfig(batch_size=50,
#                                       num_input_classes=20,
#                                       num_output_classes=2,
#                                       starting_learning_rate=0.01,
#                                       decay_steps=50,
#                                       decay_rate=0.99,
#                                       num_units=50,  # 50
#                                       train_steps=200,
#                                       keep_prop=0.5,
#                                       l2_beta=0.01)

step3_config = joint_model.StepConfig(batch_size=50,
                                      num_input_classes=20,
                                      num_output_classes=2,
                                      starting_learning_rate=0.1,
                                      decay_steps=50,
                                      decay_rate=0.96,
                                      num_units=20,
                                      train_steps=500,
                                      keep_prop=0.5,
                                      l2_beta=1.0)

model_config = joint_model.ModelConfig(step1_config=step1_config, step3_config=step3_config)


def compare_datasets():
    logdir = "test/"

    opm_set1 = "opm_set1"
    opm_set2 = "opm_set2"
    opm_set3 = "opm_set3"
    opm_set4 = "opm_set4"

    pdbtm_set1 = "pdbtm_set1"
    pdbtm_set2 = "pdbtm_set2"
    pdbtm_set3 = "pdbtm_set3"
    pdbtm_set4 = "pdbtm_set4"

    test1 = {"trainset": ["opm_set1", "opm_set2"],
             "validationset": ["opm_set3"]}

    test2 = {"trainset": ["pdbtm_set1", "pdbtm_set2"],
             "validationset": ["pdbtm_set3"]}

    test3 = {"trainset": ["opm_set1", "opm_set2", "pdbtm_set1", "pdbtm_set2"],
             "validationset": ["opm_set3"]}

    test4 = {"trainset": ["opm_set1", "opm_set2", "pdbtm_set1", "pdbtm_set2"],
             "validationset": ["pdbtm_set3"]}

    tests = [test1, test2, test3, test4]

    statistics = Statistics()

    for i, test in enumerate(tests):
        test["testset"] = test["validationset"]
        dataprovider = joint_dataprovider.Dataprovider(**test)

        runs = []
        for _ in range(3):
            m = joint_model.Model(logdir=logdir, config=model_config, dataprovider=dataprovider)

            m.train_step1()
            runs.append(m.inference())

        decoded_runs = []
        for set_lengths, set_inputs, set_targets, set_predictions in runs:
            corrected_predictions = util.numpy_step2(set_predictions)
            predictions = zip(set_lengths, set_inputs, set_targets, corrected_predictions)
            decoded_runs.append(decoder.decode_step123(predictions))

        statistics.add_model(("Test%s" % i, decoded_runs))
    statistics.print_statistics()


def test():
    statistics = Statistics()

    logdir = "test/"
    trainset = ["opm_set1", "opm_set2", "opm_set3"]
    # trainset = ["opm_set1", "opm_set2"]
    # trainset = ["opm_set1", "opm_set2", "pdbtm_set1", "pdbtm_set2"]
    validationset = testset = ["opm_set4"]
    # validationset = testset = ["opm_set3"]

    dataprovider = joint_dataprovider.Dataprovider(trainset=trainset,
                                                   validationset=validationset,
                                                   testset=testset)

    runs = []
    for i in range(1):
        m = joint_model.Model(logdir=logdir, config=model_config, dataprovider=dataprovider)

        # m.build_step1(logdir=logdir + "step1/test_model/")
        # m.build_step3(logdir=logdir + "step3/test_model/")

        start = time.time()
        # step1_logdir = m.train_step1()
        step1_train_time = time.time() - start

        start = time.time()
        # step3_logdir = m.train_step3()
        step3_train_time = time.time() - start

        step1_logdir = "test/step1/test_model/"
        # step3_logdir = "test/step3/test_model/"
        step3_logdir = None
        runs.append(m.inference(step1_logdir=step1_logdir, step3_logdir=step3_logdir))

    step1_predictions = []
    step2_predictions = []
    step3_predictions = []

    for set_lengths, set_inputs, set_targets, set_predictions, set_step2_predictions, set_step3_predictions in runs:
        predictions = zip(set_lengths, set_inputs, set_targets, set_predictions)
        step1_predictions.append(decoder.decode_step123(predictions))

        predictions = zip(set_lengths, set_inputs, set_targets, set_step2_predictions)
        step2_predictions.append(decoder.decode_step123(predictions))

        if len(set_step3_predictions) > 0:
            predictions = zip(set_lengths, set_inputs, set_targets, set_step3_predictions)
            step3_predictions.append(decoder.decode_step123(predictions))

    statistics.add_model(("Step1", step1_predictions))
    statistics.add_model(("Step2", step2_predictions))
    if len(step3_predictions) > 0:
        statistics.add_model(("Step3", step3_predictions))

    print("Training time for step 1: %s" % step1_train_time)
    print("Training time for step 3: %s" % step3_train_time)

    hmm_runs = hmm.do_hmm()
    statistics.add_model(("HMM", hmm_runs))

    # statistics.print_predictions()
    statistics.print_statistics()


if __name__ == '__main__':
    test()
    # compare_datasets()
    # hyper_params_search.do_hyper_params_search()
    # cross_validation.do_3_fold_cross_validation()
