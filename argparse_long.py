import argparse
parser = argparse.ArgumentParser(description="hello arguments there...")

parser.add_argument('--noarg', action="store_true",
                    default=False)
parser.add_argument('--witharg', action="store",
                    dest="witharg")
parser.add_argument('--witharg2',
                    dest="witharg2", type=int)

args = parser.parse_args()
print(args.witharg2)
