#!/usr/bin/env python
#from comet_ml import Experiment
from FLAlgorithms.servers.serverAbnormalDetection import AbnormalDetection as AbnormalDetectionFedRS
from FLAlgorithms.servers.serverAbnormalDetectionPG import AbnormalDetectionPG
import torch
torch.manual_seed(0)
from utils.options import args_parser

# import comet_ml at the top of your file
#                                                                                                                           
# Create an experiment with your api key:
def build_fedrs_server(
    experiment,
    device,
    dataset,
    algorithm,
    rho1,
    rho2,
    alpha,
    beta,
    num_glob_iters,
    local_epochs,
    clients,
    numusers,
    dim,
    times,
    exp_type,
    threshold,
):
    return AbnormalDetectionFedRS(
        algorithm,
        experiment,
        device,
        dataset,
        rho1,
        rho2,
        alpha,
        beta,
        num_glob_iters,
        local_epochs,
        clients,
        numusers,
        dim,
        times,
        exp_type,
        threshold=threshold,
    )


def build_pg_server(
    experiment,
    device,
    dataset,
    algorithm,
    learning_rate,
    ro,
    num_glob_iters,
    local_epochs,
    clients,
    numusers,
    dim,
    times,
    exp_type,
    threshold,
):
    return AbnormalDetectionPG(
        algorithm,
        experiment,
        device,
        dataset,
        learning_rate,
        ro,
        num_glob_iters,
        local_epochs,
        clients,
        numusers,
        dim,
        times,
        exp_type,
        threshold=threshold,
    )


def main(
    experiment,
    dataset,
    algorithm,
    batch_size,
    rho1,
    rho2,
    alpha,
    beta,
    num_glob_iters,
    local_epochs,
    clients,
    numusers,
    dim,
    threshold,
    times,
    gpu,
    exp_type,
    learning_rate,
    ro,
):
    
    # Get device status: Check GPU or CPU
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")
    data = dataset
    if algorithm in {"FedPG", "FedPE"}:
        print("fedPG Server is built")
        server = build_pg_server(
            experiment,
            device,
            data,
            algorithm,
            learning_rate,
            ro,
            num_glob_iters,
            local_epochs,
            clients,
            numusers,
            dim,
            times,
            exp_type,
            threshold,
        )
    else:
        print("fedRS Server is built")
        server = build_fedrs_server(
            experiment,
            device,
            data,
            algorithm,
            rho1,
            rho2,
            alpha,
            beta,
            num_glob_iters,
            local_epochs,
            clients,
            numusers,
            dim,
            times,
            exp_type,
            threshold,
        )
    server.train()

if __name__ == "__main__":
    args = args_parser()
    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("rho1                 : {}".format(args.rho1))
    print("rho2                 : {}".format(args.rho2))
    print("alpha                : {}".format(args.alpha))
    print("beta                 : {}".format(args.beta))
    print("Subset of users      : {}".format(args.subusers))
    print("Number of global rounds       : {}".format(args.num_global_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    # print("Dataset       : KDD")
    print("Dataset       : {}".format(args.dataset))
    print("=" * 80)

    if(args.commet):
        # Create an experiment with your api key:
        experiment = Experiment(
            api_key="VtHmmkcG2ngy1isOwjkm5sHhP",
            project_name="multitask-for-test",
            workspace="federated-learning-exp",
        )

        hyper_params = {
            "dataset":args.dataset,
            "algorithm" : args.algorithm,
            "batch_size":args.batch_size,
            "rho1": args.rho1,
            "rho2": args.rho2,
            "alpha": args.alpha,
            "beta": args.beta,
            "dim" : args.dim,
            "num_glob_iters":args.num_global_iters,
            "local_epochs":args.local_epochs,
            "clients":args.clients,
            "numusers": args.subusers,
            "threshold": args.threshold,
            "times" : args.times,
            "gpu": args.gpu,
            "cut-off": args.cutoff
        }
        
        experiment.log_parameters(hyper_params)
    else:
        experiment = 0

    main(
        experiment= experiment,
        dataset=args.dataset,
        algorithm = args.algorithm,
        batch_size=args.batch_size,
        rho1 = args.rho1,
        rho2 = args.rho2,
        alpha = args.alpha,
        beta = args.beta,
        num_glob_iters=args.num_global_iters,
        local_epochs=args.local_epochs,
        clients = args.clients,
        numusers = args.subusers,
        dim = args.dim,
        threshold = args.threshold,
        times = args.times,
        gpu=args.gpu,
        exp_type=args.exp_type,
        learning_rate=args.learning_rate,
        ro=args.ro,
        )


