import argparse

from foundwsr.experiment import Experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="signal", type=str, help="name of models")
    parser.add_argument("--dataset", "-d", default="RML2016.10a", type=str, help="list of datasets splited with #")
    parser.add_argument("--gpu", "-g", default="0", type=str, help="-1 means cpu")
    parser.add_argument("--compile", action="store_true", help="compile model")

    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--length", default=128, type=int)
    parser.add_argument("--num_classes", default=11, type=int)
    parser.add_argument("--max_step", default=3000, type=int)
    parser.add_argument("--timestep", default=4, type=int)
    parser.add_argument("--ratio", default=0.414, type=float)
    parser.add_argument("--min_noise", default=5.47e-7, type=float)
    parser.add_argument("--max_noise", default=0.0072, type=float)

    args = parser.parse_args()

    dataset = args.dataset.split("#")
    gpu = list(map(int, args.gpu.split("#")))
    if len(gpu) == 1:
        gpu = gpu[0]
    elif any(x < 0 for x in gpu):
        raise ValueError("Negative numbers should not appear in the GPU list!")
    
    model_path = "xxx"
    
    experiment = Experiment(model=args.model, dataset=dataset, task="prob", gpu=gpu, batch_size=args.batch_size,
                            load_from_pretrained=True, compile_flag=args.compile,
                            evaluate_interval=1, max_step=args.max_step, ratio=args.ratio, signal_length=args.length,
                            num_classes = args.num_classes, min_noise=args.min_noise, max_noise=args.max_noise, 
                            timestep=args.timestep, model_path=model_path)

    experiment.run()