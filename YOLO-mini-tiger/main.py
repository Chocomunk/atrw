from __future__ import print_function
import os
import boto3
from subprocess import call


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='YOLO Training')
    parser.add_argument('--no-cuda', type=bool, default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', type=bool, default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', type=bool, default=False,
                        help='For Saving the current Model')
    parser.add_argument('--out-bucket', type=str, default='calvinandpogs-ee148',
                        help='For Saving the current Model')
    parser.add_argument('--out-prefix', type=str, default='detection/out/',
                        help='For Saving the current Model')

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--out-data', type=str, default=os.environ['SM_CHANNEL_OUTDATA'])

    args = parser.parse_args()

    os.chmod('train_darknet.sh', 0o755)
    rc = call("./train_darknet.sh")

    s3 = boto3.resource('s3')
    s3.bucket(args.out_bucket).upload_file(
        "darknet/backup/yolo-mini-tiger_final.weights", 
        "{}yolo-mini-tiger.weights".format(args.out_prefix))

    # ... train `model`, then save it to `model_dir`
    # with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
    #     torch.save(model.state_dict(), f)


if __name__ == '__main__':
    main()