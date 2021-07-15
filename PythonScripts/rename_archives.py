import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help = "Base directory for agent models.")
    parser.add_argument("new_name", type=str, help="What to rename all 'archive.zip' archives to")
    args = parser.parse_args()
    print(args)

    for subdir, dirs, files in os.walk(args.model_dir):
        for dir in dirs:
            if dir[-1] != '/' or dir[-1] != '\\':
                dir += '/'
            if os.path.exists(subdir+dir+"archive.zip"):
                os.system("mv " + subdir+dir+"archive.zip " + subdir+dir+args.new_name)