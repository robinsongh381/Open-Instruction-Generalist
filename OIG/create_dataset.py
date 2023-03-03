import os
from argparse import ArgumentParser
from src.dataset import AVAILABLE_TASKS_DICT


def main():    
    parser = ArgumentParser()
    parser.add_argument("--task_name") # , choices=AVAILABLE_TASKS_DICT.keys())
    parser.add_argument("--output_path", default=None)    
    args = parser.parse_args()
    
    assert args.task_name in AVAILABLE_TASKS_DICT, f"args.task_name should be one of {list(AVAILABLE_TASKS_DICT.keys())}"
    
    if args.output_path is None:
        args.output_path = f'processed/{args.task_name}.json'
        os.makedirs("processed", exist_ok=True)        
    else:
        output_dir = "/".join(args.output_path.split('/')[:-1])
        os.makedirs(output_dir)
    
    data_args = {
        'output_path': args.output_path        
    }
    
    processor = AVAILABLE_TASKS_DICT[args.task_name](**data_args)
    processor()
    print(f'Process done. Result is saved to {args.output_path}')
    
    
if __name__ == "__main__":
    main()