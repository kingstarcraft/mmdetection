import sys
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('ckpt_root', help='checkpoint file')
    parser.add_argument('output', help='output file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    args = parser.parse_args()
    return args


def main():
    root = sys.path[0].replace('\\', '/')
    args = parse_args()
    stems = []
    for filename in os.listdir(args.ckpt_root):
        if filename.startswith('epoch') and filename.endswith('.pth'):
            try:
                epoch = int(filename[6:-4])
                stems.append(epoch)
            except:
                raise
    for stem in sorted(stems):
        ckpt = args.ckpt_root + f'/epoch_{stem}.pth'
        os.system(f'python {root}/test.py {args.config} {ckpt} --work-dir {args.output} '
                  f'--gpu {args.gpu} '
                  f'--eval bbox --eval-options classwise=True '
                  f'--dect threshold=0.9 size=1024,1024 overlap=100')


if __name__ == '__main__':
    main()
