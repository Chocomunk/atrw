import os
import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input file to sample from')
    parser.add_argument('--output', type=str, help='File to store output')
    parser.add_argument('--frac', type=float, help='Proportion of input to sample')
    args = parser.parse_args()
    
    # Check files and directories
    assert os.path.isfile(args.input), "Input is not a valid file!"
    
    out_dir = os.path.dirname(args.output)
    if out_dir != '' and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        
    # Randomly sample from input file
    with open(args.input) as f:
        orig_lines = f.readlines()
        lines = random.sample(orig_lines, int(len(orig_lines) * args.frac))
        
    # Write lines to output file
    with open(args.output, 'w') as f:
        f.writelines(lines)
    
if __name__ == '__main__':
    main()
