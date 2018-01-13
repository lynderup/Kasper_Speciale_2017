import dataprovider.joint_dataprovider as joint_dataprovider
import model.joint_model as joint_model


def do_3_fold_cross_validation(config=None, logdir=None):

    if logdir is None:
        logdir = "test/"

    sets = ["opm_set1", "opm_set2", "opm_set3"]

    runs = []

    print("Doing 3-fold cross validation")
    for i in range(3):
        print("Fold %i" % i)
        trainset = sets[:i] + sets[i + 1:]
        validationset = testset = sets[i:i + 1]

        dataprovider = joint_dataprovider.Dataprovider(trainset=trainset,
                                                       validationset=validationset,
                                                       testset=testset)

        m = joint_model.Model(logdir=logdir, config=config, dataprovider=dataprovider)

        # step1_logdir = m.train_step1()
        step3_logdir = m.train_step3()

        step1_logdir = "test/step1/test_model/"
        # step3_logdir = logdir + "step3/test_model/"
        runs.append(m.inference(step1_logdir=step1_logdir, step3_logdir=step3_logdir))

    return runs
