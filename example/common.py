import argparse

from foundwsr.experiment import Experiment
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="IQFormer", type=str, help="name of models")
    parser.add_argument("--task", "-t", default="amc", type=str, help="name of task")
    parser.add_argument("--dataset", "-d", default="RML2016.10a", type=str, help="list of datasets splited with #")
    parser.add_argument("--gpu", "-g", default="0", type=str, help="-1 means cpu")
    parser.add_argument("--load_from_pretrained", action="store_true", help="load model from the checkpoint")
    parser.add_argument("--compile", action="store_true", help="compile model")

    args = parser.parse_args()

    dataset = args.dataset.split("#")
    gpu = list(map(int, args.gpu.split("#")))
    if len(gpu) == 1:
        gpu = gpu[0]
    elif any(x < 0 for x in gpu):
        raise ValueError("Negative numbers should not appear in the GPU list!")
    experiment = Experiment(model=args.model, dataset=dataset, task=args.task, gpu=gpu,
                            load_from_pretrained=args.load_from_pretrained, compile_flag=args.compile,
                            evaluate_interval=1)

    experiment.run()